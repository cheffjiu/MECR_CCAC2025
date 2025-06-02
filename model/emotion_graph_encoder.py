import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm


class EmotionGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=256, heads=4, dropout=0.1):
        """
        图神经网络模块，用于处理对话图并提取结构性情感变化表示。

        Args:
            in_dim (int): 输入节点特征维度（如融合后的多模态特征）
            hidden_dim (int): 中间隐层维度
            out_dim (int): 输出节点特征维度
            heads (int): 多头注意力的头数
            dropout (float): Dropout 概率
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels=self.in_dim,
            out_channels=self.hidden_dim // self.heads,
            heads=self.heads,
            concat=True,
            edge_dim=1,
        )
        self.norm1 = GraphNorm(self.hidden_dim)
        self.dropout1 = nn.Dropout(self.dropout)

        self.conv2 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.out_dim,
            heads=1,
            concat=False,
            edge_dim=1,
        )
        self.norm2 = GraphNorm(self.out_dim)
        self.dropout2 = nn.Dropout(self.dropout)

        # 映射到结构特征向量
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim, 612),
            nn.GELU(),
            nn.LayerNorm(612),
            nn.Linear(612, 1024),
        )

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Batch): 图结构数据，包含：
                - data.x: 节点特征 (N, in_dim)
                - data.edge_index: 边索引 (2, E)
                - data.edge_attr: 边属性 (E, 1)
                - data.super_idx: 每个图中超级节点索引 (B,)

        Returns:
            h_change (Tensor): 超级节点向量 (B, 1024)
            h_all (Tensor): 所有节点的图编码特征 (N, out_dim)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 第一层 GAT + Norm + Dropout
        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h, data.batch)
        h = F.elu(h)
        h = self.dropout1(h)

        # 第二层 GAT
        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h, data.batch)
        h = F.elu(h)
        h = self.dropout2(h)

        # 超级节点向量提取（支持 batched 图）
        super_idx = data.super_idx  # shape (B,)
        h_change = h[super_idx]  # shape (B, out_dim)

        # MLP 提升维度
        h_change = self.mlp(h_change)

        return h_change, h
