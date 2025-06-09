import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm


class EmotionGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=256, num_heads=4, dropout=0.1, num_gnn_layers=4):
        """
        图神经网络模块，用于处理对话图并提取结构性情感变化表示。
        改造为多层 (num_gnn_layers) Transformer-like 结构。

        Args:
            in_dim (int): 输入节点特征维度
            hidden_dim (int): 中间隐层维度（也是 GNN 层的输出维度，除最后一层）
            out_dim (int): 最终输出节点特征维度（最后一层 GNN 的输出维度）
            num_heads (int): 多头注意力的头数
            dropout (float): Dropout 概率
            num_gnn_layers (int): GNN 层的总数，默认为 4
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers

        # 确保 hidden_dim 是 num_heads 的倍数，以便 GATConv 可以拼接多头输出
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be a multiple of num_heads ({self.num_heads})"

        # GNN 层的列表
        self.gnn_layers = nn.ModuleList()
        # 残差连接的投影层列表 (用于处理维度不匹配的情况)
        self.res_projs = nn.ModuleList()
        # 注意力子层的归一化和 Dropout
        self.norm_attns = nn.ModuleList()
        self.dropout_attns = nn.ModuleList()
        # 前馈网络 (FFN) 子层
        self.ffns = nn.ModuleList()
        # FFN 子层的归一化和 Dropout
        self.norm_ffns = nn.ModuleList()
        self.dropout_ffns = nn.ModuleList()

        # 构建 GNN 层 (Transformer Block 结构)
        for i in range(self.num_gnn_layers):
            current_in_dim = self.in_dim if i == 0 else self.hidden_dim
            current_out_dim = self.hidden_dim

            # 最后一层 GNN 的特殊处理：输出维度为 out_dim，单头，不拼接
            if i == self.num_gnn_layers - 1:
                self.gnn_layers.append(
                    GATConv(
                        in_channels=current_in_dim,
                        out_channels=self.out_dim, # 最后一层输出到 out_dim
                        heads=1, # 最后一层使用单头
                        concat=False, # 不拼接
                        edge_dim=1,
                        dropout=self.dropout # GATConv 内部的注意力 Dropout
                    )
                )
                # FFN 的输入输出维度也随之改变
                ffn_dim = self.out_dim
            else:
                self.gnn_layers.append(
                    GATConv(
                        in_channels=current_in_dim,
                        out_channels=self.hidden_dim // self.num_heads,
                        heads=self.num_heads,
                        concat=True, # 拼接多头
                        edge_dim=1,
                        dropout=self.dropout
                    )
                )
                ffn_dim = self.hidden_dim

            # 残差连接投影层：如果当前输入维度和 GNN 输出维度不同，则需要投影
            if current_in_dim != ffn_dim:
                self.res_projs.append(nn.Linear(current_in_dim, ffn_dim))
            else:
                self.res_projs.append(nn.Identity()) # 维度相同则无需投影

            # 注意力子层的归一化和 Dropout
            self.norm_attns.append(GraphNorm(ffn_dim))
            self.dropout_attns.append(nn.Dropout(self.dropout))

            # 前馈网络 (FFN) 子层
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(ffn_dim, ffn_dim * 2), # 常见的 FFN 内部放大两倍
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ffn_dim * 2, ffn_dim)
                )
            )
            # FFN 子层的归一化和 Dropout
            self.norm_ffns.append(GraphNorm(ffn_dim))
            self.dropout_ffns.append(nn.Dropout(self.dropout))

        # MLP 提升维度：从 GNN 最终输出维度 (out_dim) 到 1024
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512), # LayerNorm for general features, not graph specific
            nn.Linear(512, 1024),
        )

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Batch): 图结构数据，包含：
                - data.x: 节点特征 (N, in_dim)
                - data.edge_index: 边索引 (2, E)
                - data.edge_attr: 边属性 (E, 1)
                - data.batch: 节点到批次索引的映射 (N,)
                - data.super_idx: 每个图中超级节点索引 (B,)

        Returns:
            h_change (Tensor): 超级节点向量 (B, 1024)
            h_all (Tensor): 所有节点的图编码特征 (N, out_dim)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch # 用于 GraphNorm

        # 迭代所有 GNN 层 (Transformer Block)
        for i in range(self.num_gnn_layers):
            h_current_layer_input = x # 保存当前层的输入，用于残差连接

            # 如果维度不匹配，则对输入进行投影，用于残差连接
            res_val = self.res_projs[i](h_current_layer_input)

            # --- 1. Graph Attention Sub-layer (Pre-Norm) ---
            # 先归一化再进行 GATConv
            h_attn_normed = self.norm_attns[i](h_current_layer_input, batch)
            h_attn_output = self.gnn_layers[i](h_attn_normed, edge_index, edge_attr)
            h_attn_output = F.elu(h_attn_output)
            h_attn_output = self.dropout_attns[i](h_attn_output)
            # 添加残差连接
            h_after_attn = res_val + h_attn_output

            # --- 2. Feed-Forward Network Sub-layer (Pre-Norm) ---
            # 先归一化再进行 FFN
            h_ffn_normed = self.norm_ffns[i](h_after_attn, batch)
            h_ffn_output = self.ffns[i](h_ffn_normed)
            h_ffn_output = F.elu(h_ffn_output) # FFN 内部可能有激活，这里额外再加一层
            h_ffn_output = self.dropout_ffns[i](h_ffn_output)
            # 添加残差连接
            x = h_after_attn + h_ffn_output # 将当前块的输出作为下一块的输入

        # 经过所有 GNN 层后，x 包含了所有节点的最终图编码特征 (N, out_dim)
        h_all = x

        # 超级节点向量提取（支持 batched 图）
        super_idx = data.super_idx
        h_change = h_all[super_idx] # shape (B, out_dim)

        # MLP 提升维度到 1024
        h_change = self.mlp(h_change)

        return h_change, h_all