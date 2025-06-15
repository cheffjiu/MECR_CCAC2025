import torch
import torch.nn as nn
from feature_fusion import CrossModalAttention
from emotion_graph_encoder import EmotionGraphEncoder


class MultimodalEmotionGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        gnn_dim,
        # GNN模块参数
        gnn_in_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_out_dim: int = 256,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.1,
    ):
        super().__init__()

        # 保存原始维度信息
        self.in_dim = in_dim
        self.gnn_dim = gnn_dim

        # 1.把原始特征维度映射到GNN输入维度
        self.fusion_dim_to_gnn_dim = nn.Sequential(
            nn.Linear(self.in_dim, self.gnn_dim),
            nn.GELU(),
            nn.LayerNorm(gnn_in_dim),
        )
        # 2. 实例化图编码器模块
        # GNN的输入维度就是融合后的维度 d_fusion
        self.gnn_module = EmotionGraphEncoder(
            in_dim=self.gnn_dim,
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
        # 映射到GNN输入维度
        fused_node_features = self.fusion_dim_to_gnn_dim(original_node_features)
        # 3. 更新图数据中的节点特征为融合后的特征
        data.x = fused_node_features
        # 4. 将更新后的图送入GNN编码器进行图级别的推理
        h_change, h_all = self.gnn_module(data)
        # data.x = original_node_features
        # 5. 返回最终结果
        return h_change, h_all
