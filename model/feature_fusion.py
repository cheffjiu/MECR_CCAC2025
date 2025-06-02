import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, d_fusion):
        super().__init__()
        self.d_fusion = d_fusion
        self.gate = nn.Sequential(
            nn.Linear(2 * self.d_fusion, self.d_fusion), nn.Sigmoid()
        )
        self.fc = nn.Linear(self.d_fusion, self.d_fusion)
        self.norm = nn.LayerNorm(self.d_fusion)

    def forward(self, t_feat: torch.Tensor, v_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_feat (torch.Tensor): 文本特征，shape = [B, d_fusion]
            v_feat (torch.Tensor): 视觉特征，shape = [B, d_fusion]
        Returns:
            fused (torch.Tensor): 融合特征，shape = [B, d_fusion]
        """
        combined = torch.cat([t_feat, v_feat], dim=-1)  # [B, 2*d_fusion]
        gate = self.gate(combined)  # [B, d_fusion]
        fused = gate * t_feat + (1 - gate) * v_feat  # 门控加权融合
        fused = self.fc(fused)
        return self.norm(fused + t_feat + v_feat)  # 残差连接


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        d_t: int,
        d_v: int,
        d_fusion: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_t = d_t
        self.d_v = d_v
        self.d_fusion = d_fusion
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # 文本、视觉投影
        self.text_proj = nn.Sequential(
            nn.Linear(self.d_t, self.d_fusion), nn.GELU(inplace=True)
        )
        self.vis_proj = nn.Sequential(
            nn.Linear(self.d_v, self.d_fusion), nn.GELU(inplace=True)
        )

        # 交叉注意力层
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "t2v_attn": nn.MultiheadAttention(
                            self.d_fusion,
                            self.num_heads,
                            dropout=self.dropout,
                            batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(self.d_fusion),
                        "v2t_attn": nn.MultiheadAttention(
                            self.d_fusion,
                            self.num_heads,
                            dropout=self.dropout,
                            batch_first=True,
                        ),
                        "norm2": nn.LayerNorm(self.d_fusion),
                        "dropout": nn.Dropout(self.dropout),
                    }
                )
                for _ in range(self.num_layers)
            ]
        )

        # 融合头
        self.fusion_head = GatedFusion(self.d_fusion)

    def forward(self, t_feats: torch.Tensor, v_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_feats (torch.Tensor): 文本特征，shape = [B, d_t]
            v_feats (torch.Tensor): 视觉特征，shape = [B, d_v]
        Returns:
            fused_output (torch.Tensor): 融合特征，shape = [B, d_fusion]
        """
        # 输入为 [B, D]
        t_proj = self.text_proj(t_feats)  # [B, d_fusion]
        v_proj = self.vis_proj(v_feats)  # [B, d_fusion]

        # 加入 batch 维度 → [1, B, d_fusion]，因为 MultiheadAttention 需要 batch_first=True
        t_proj = t_proj.unsqueeze(0)
        v_proj = v_proj.unsqueeze(0)

        for layer in self.layers:
            # 文本到视觉的注意力
            t_attn, _ = layer["t2v_attn"](query=t_proj, key=v_proj, value=v_proj)
            t_proj = layer["norm1"](t_proj + layer["dropout"](t_attn))

            # 视觉到文本的注意力
            v_attn, _ = layer["v2t_attn"](query=v_proj, key=t_proj, value=t_proj)
            v_proj = layer["norm2"](v_proj + layer["dropout"](v_attn))

        # 去除 batch 维度 → [B, d_fusion]
        t_proj = t_proj.squeeze(0)
        v_proj = v_proj.squeeze(0)

        fused_output = self.fusion_head(t_proj, v_proj)
        return fused_output
