from peft.tuners.lora import Linear as LoRALinear
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
class HChangeCrossAttention(nn.Module):
    def __init__(self, hidden_dim=4096, h_change_dim=4096, num_heads=8):  # 修改输入维度
        super().__init__()
        self.query_proj = LoRALinear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = LoRALinear(h_change_dim, hidden_dim, bias=False)  # 输入维度匹配
        self.value_proj = LoRALinear(h_change_dim, hidden_dim, bias=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, h_change):
        if h_change is None:
            return hidden_states

        B, L, D = hidden_states.shape
        query = self.query_proj(hidden_states)
        h_change = h_change.unsqueeze(1)
        key = self.key_proj(h_change)
        value = self.value_proj(h_change)

        out, _ = self.attn(query, key, value)
        return self.norm(hidden_states + out)
