import torch
import torch.nn as nn
from feature_fusion import CrossModalAttention
from emotion_graph_encoder import EmotionGraphEncoder


class MultimodalEmotionGNN(nn.Module):
    def __init__(
        self,
        # 原始特征维度
        t_feat_dim: int,
        v_feat_dim: int,
        # 融合模块参数
        d_fusion: int,
        fusion_heads: int,
        fusion_layers: int,
        fusion_dropout: float = 0.1,
        # GNN模块参数
        gnn_in_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_out_dim: int = 256,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.1,
    ):
        """
        整合了多模态融合与图神经网络的端到端模型。

        Args:
            t_feat_dim (int): 原始文本特征维度
            v_feat_dim (int): 原始视觉特征维度
            d_fusion (int): 融合后的特征维度
            fusion_heads (int): 交叉注意力头数
            fusion_layers (int): 交叉注意力层数
            ... (其他GNN参数)
        """
        super().__init__()

        # 保存原始维度信息，用于特征分离
        self.t_feat_dim = t_feat_dim
        self.v_feat_dim = v_feat_dim

        # 1. 实例化多模态融合模块
        self.fusion_module = CrossModalAttention(
            d_t=t_feat_dim,
            d_v=v_feat_dim,
            d_fusion=d_fusion,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            dropout=fusion_dropout,
        )
        self.fusion_dim_to_gnn_dim=nn.Sequential(
            nn.Linear(d_fusion+1, gnn_in_dim),
            nn.GELU(),
            nn.LayerNorm(gnn_in_dim),
        )
        # 2. 实例化图编码器模块
        # GNN的输入维度就是融合后的维度 d_fusion
        self.gnn_module = EmotionGraphEncoder(
            in_dim=d_fusion,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            num_heads=gnn_heads,
            dropout=gnn_dropout,
        )

    def forward(self, data):
        """
        端到端的前向传播。

        Args:
            data (torch_geometric.data.Batch): 图数据。
                data.x 是 t_feats 和 v_feats和位置编码v_pos简单拼接的结果。

        Returns:
            h_change (Tensor): 超级节点向量 (B, 1024)
            h_all (Tensor): 所有节点的图编码特征 (N, gnn_out_dim)
        """
        
        # 原始节点特征，shape: [num_total_nodes, t_feat_dim + v_feat_dim + v_pos_dim]
        original_node_features = data.x

        # 1. 从拼接的特征中分离出文本和视觉特征
        # 节点数 N 此时可以看作是融合模块的批大小 B
        node_t_feats = original_node_features[:, : self.t_feat_dim]
        node_v_feats = original_node_features[:, self.t_feat_dim : -1]
        node_v_pos = original_node_features[:, -1]
        # print(f"node_t_feats: {node_v_pos.shape}")
        
        # 2. 对每个节点的特征进行多模态融合
        # 输入: [N, d_t], [N, d_v] -> 输出: [N, d_fusion]
        fused_node_features = self.fusion_module(node_t_feats, node_v_feats)
        # print(f"fused_node_features: {fused_node_features.shape}")
        # 拼接上位置编码
        fused_node_features = torch.cat([fused_node_features, node_v_pos.unsqueeze(-1)], dim=-1)
        # 映射到GNN输入维度
        fused_node_features = self.fusion_dim_to_gnn_dim(fused_node_features)
        # 3. 更新图数据中的节点特征为融合后的特征
        data.x = fused_node_features
        # 4. 将更新后的图送入GNN编码器进行图级别的推理
        h_change, h_all = self.gnn_module(data)
        data.x = original_node_features

        # 5. 返回最终结果
        return h_change, h_all
