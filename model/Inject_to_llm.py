import torch
import torch.nn as nn
import torch.nn.functional as F


class InjectionModule(nn.Module):
    """
    优化后的特征注入模块，负责将 GNN 输出特征与 LLM 隐藏状态融合
    通过交叉注意力和门控机制实现高效特征融合
    """

    def __init__(self, d_gnn: int, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_gnn (int): GNN 输出特征维度
            d_model (int): LLM 隐藏状态维度
            n_heads (int): 交叉注意力头数
            dropout (float): Dropout 概率
        """
        super().__init__()
        self.d_gnn = d_gnn
        self.dropout = dropout

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除")

        self.d_model = d_model
        self.n_heads = n_heads

        # GNN 特征投影层
        self.h_change_proj = nn.Sequential(
            nn.Linear(self.d_gnn, self.d_model), nn.GELU()
        )

        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        # 门控融合机制
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model), nn.Sigmoid()
        )

        # 归一化层
        self.norm_h_change = nn.LayerNorm(self.d_model)
        self.norm_cross_attn = nn.LayerNorm(self.d_model)
        self.norm_output = nn.LayerNorm(self.d_model)

        # Dropout 层
        self.dropout = nn.Dropout(self.dropout)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """使用 Xavier 初始化提升收敛性"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "gate" in name:
                    nn.init.xavier_uniform_(
                        module.weight, gain=nn.init.calculate_gain("sigmoid")
                    )
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, h_llm: torch.Tensor, h_change: torch.Tensor | None
    ) -> torch.Tensor:
        """
        融合 GNN 特征和 LLM 隐藏状态

        Args:
            h_llm: LLM 最后一层隐藏状态 (B, T, d_model)
            h_change: GNN 输出特征 (B, d_gnn) 或 None

        Returns:
            融合后的隐藏状态 (B, T, d_model)
        """
        # 处理 h_change 为 None 的情况
        if h_change is None:
            print("未进行注入，直接返回原始 LLM 输出")
            return h_llm

        # 维度验证
        if h_llm.dim() != 3:
            raise ValueError(f"h_llm 应为 3D 张量 (B, T, D), 实际维度: {h_llm.dim()}")

        if h_change.dim() != 2:
            raise ValueError(
                f"h_change 应为 2D 张量 (B, D_gnn), 实际维度: {h_change.dim()}"
            )
        # print(f"h_change.dtype: {h_change.dtype}, h_llm.dtype: {h_llm.dtype}")
        # 确保张量在相同设备和数据类型上

        h_change = h_change.to(device=h_llm.device, dtype=h_llm.dtype)
        # print(f"h_change.dtype: {h_change.dtype}, h_llm.dtype: {h_llm.dtype}")
        # 1. 投影 GNN 特征到 LLM 空间
        h_change_proj = self.h_change_proj(h_change)  # (B, d_model)
        h_change_proj = self.norm_h_change(h_change_proj)

        # 2. 交叉注意力机制
        # 扩展为序列形式 (B, 1, d_model)
        h_change_seq = h_change_proj.unsqueeze(1)

        # 注意力计算
        attn_output, _ = self.cross_attention(
            query=h_llm,
            key=h_change_seq,
            value=h_change_seq,
            need_weights=False,  # 不需要注意力权重，节省内存
        )

        # 残差连接 + 归一化
        attn_output = self.norm_cross_attn(h_llm + self.dropout(attn_output))

        # 3. 门控融合机制
        combined = torch.cat([h_llm, attn_output], dim=-1)  # (B, T, 2*d_model)
        gate = self.gate_proj(combined)  # (B, T, d_model)

        # 门控融合
        fused_output = gate * attn_output + (1 - gate) * h_llm

        # 输出归一化
        output = self.norm_output(fused_output)

        # 调试信息

        # print(f"注入完成: LLM形状 {h_llm.shape}, GNN形状 {h_change.shape}")
        # print(f"注意力输出方差: {attn_output.var().item():.4f}")
        # print(f"门控均值: {gate.mean().item():.4f}")

        return output
