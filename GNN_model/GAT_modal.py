import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm


class EmotionGraphEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim=256, out_dim=256, heads=4, device="cpu"):
        super().__init__()
        # 在层定义后，统一调用 .to(device) 或在初始化时传入 device
        # 推荐在 __init__ 结束时调用 self.to(device) 来统一管理
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, edge_dim=1)
        self.norm1 = GraphNorm(hidden_dim)

        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, edge_dim=1)
        self.norm2 = GraphNorm(out_dim)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, 612),
            nn.GELU(),
            nn.LayerNorm(612),
            nn.Linear(612, 1024),
        )

        self.device = device  # 存储设备信息
        self.to(device)  # 将整个模块及其子模块移动到指定设备

    def forward(self, data):
        # 确保输入数据 data.x, data.edge_index, data.edge_attr 都在正确的设备上
        # collate_fn 应该已经处理了这个问题，但在这里也可以再次确认或显式移动
        x, edge_index, edge_attr = (
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.edge_attr.to(self.device),
        )

        # Super node index 也是 PyG Data 对象的一部分，需要确保其在正确设备
        # 如果 super_idx 是一个 tensor，它也应该在同一设备上
        super_idx = data.super_idx.to(self.device)

        h = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        h = F.elu(self.norm2(self.conv2(h, edge_index, edge_attr)))

        h_change = h[super_idx]  # 使用移动到设备的 super_idx
        h_change = self.mlp(h_change)
        print(f"h_change.shape: {h_change.shape}, h.shape: {h.shape}")
        return h_change, h
