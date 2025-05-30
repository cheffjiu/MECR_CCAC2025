import torch
import torch.nn as nn
import torch.nn.functional as F


class InjectionModule(nn.Module):
    """
    负责将 GNN 输出特征 (h_change) 与 LLM 的最终隐藏状态 (h) 通过交叉注意力融合，
    并加入门控机制来控制 h_change 的影响。
    融合后的输出 `m` 将传递给 Qwen 的 MLP 层（或 lm_head）。
    """

    def __init__(self, d_gnn: int, d_model: int, n_heads: int):
        """
        Args:
            d_gnn (int): GNN 输出特征 h_change 的维度。
            d_model (int): LLM 隐藏状态的维度 (即 Query, Key, Value 的维度)。
            n_heads (int): 交叉注意力的头数。
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) for MultiheadAttention."
            )

        # 将 GNN 输出特征从 d_gnn 投影到 d_model
        self.h_change_proj = nn.Linear(d_gnn, d_model)
        self.norm_h_change_proj = nn.LayerNorm(d_model)

        # PyTorch 的 MultiheadAttention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm_cross_attn = nn.LayerNorm(d_model)
        self.dropout_cross_attn = nn.Dropout(0.1)

        # 门控机制
        self.fusion_gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), 
            nn.Sigmoid()
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, h_llm: torch.Tensor, h_change: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Args:
            h_llm (torch.Tensor): LLM 的最后一层隐藏状态，形状为 (B, T, d_model)。
            h_change (torch.Tensor | None): GNN 输出特征，形状为 (B, d_gnn)。
        Returns:
            torch.Tensor: 融合 h_change 后的隐藏状态 `m`，形状为 (B, T, d_model)。
        """
        if h_change is None:
            print("未进行注入，直接返回原始 LLM 输出。")
            return h_llm

        B, T, D_model = h_llm.size()
        
        # 确保数据类型一致
        h_change = h_change.to(dtype=h_llm.dtype, device=h_llm.device)

        # 1. 投影 h_change 到 LLM 隐藏状态维度
        h_change_proj = self.h_change_proj(h_change)  # (B, d_model)
        h_change_proj = self.norm_h_change_proj(h_change_proj)

        # 2. 扩展为序列形式进行交叉注意力
        # 使用更稳定的扩展方式
        h_change_seq = h_change_proj.unsqueeze(1)  # (B, 1, d_model)
        
        # 交叉注意力：LLM隐藏状态关注GNN特征
        attn_output, _ = self.cross_attention(
            query=h_llm,           # (B, T, d_model)
            key=h_change_seq,      # (B, 1, d_model)
            value=h_change_seq     # (B, 1, d_model)
        )
        
        attn_output = self.dropout_cross_attn(attn_output)
        attn_output = self.norm_cross_attn(h_llm + attn_output)

        # 3. 门控机制融合
        gating_input = torch.cat([h_llm, attn_output], dim=-1)  # (B, T, 2*d_model)
        gate_values = self.fusion_gate_proj(gating_input)       # (B, T, d_model)

        # 最终融合
        fused_output = gate_values * attn_output + (1 - gate_values) * h_llm
        fused_results = self.final_norm(fused_output)
        
        print (f"h_llm 形状: {h_llm.shape}, h_change 形状: {h_change.shape}, fused_output 形状: {fused_output.shape}")
        print("成功注入 h_change 到 LLM 隐藏状态。")

        return fused_results