# fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, t_feat, v_feat):
        combined = torch.cat([t_feat, v_feat], dim=-1)
        gate = self.gate(combined)
        fused = gate * t_feat + (1 - gate) * v_feat
        fused = self.fc(fused)
        # 残差连接 (fused + t_feat) 会使最终输出偏向文本特征
        return self.norm(fused + t_feat)

class CrossModalAttention(nn.Module):
    def __init__(self, d_t=768, d_v=512, d_model=256, num_heads=4, dropout=0.1, num_layers=3, device='cpu'):
        super().__init__()
        
        self.text_proj = nn.Sequential(
            nn.Linear(d_t, d_model),
            nn.LayerNorm(d_model)
        )

        self.vis_proj = nn.Sequential(
            nn.Linear(d_v, d_model),
            nn.LayerNorm(d_model)
        )
        
        # vis_dummy 是一个可学习的参数，代表缺失的视觉特征。
        # 初始化为零，但模型可以学习一个更合适的“缺失”状态的表征。
        self.vis_dummy = nn.Parameter(torch.zeros(d_model, dtype=torch.float32, device=device))
        
        self.device = device 

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                't2v_attn': nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'v2t_attn': nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True),
                'norm2': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        
        self.fusion_head = GatedFusion(d_model)
        self.to(device) # 将所有模块移动到指定设备

    def forward(self, t_feats, v_feats, vis_mask=None, text_mask=None):
        t_feats = t_feats.to(self.device)
        # vis_mask 对应位置为0，则 v_feats 中对应样本的特征已经是全零张量
        v_feats = v_feats.to(self.device) 
        
        t_proj = self.text_proj(t_feats)
        # 对原始视觉特征（可能包含全零张量）进行投影
        v_proj_initial = self.vis_proj(v_feats) 
        
        if vis_mask is not None:
            vis_mask = vis_mask.to(self.device) # vis_mask: True 代表存在，False 代表缺失
            mask_expanded = vis_mask.unsqueeze(-1).bool() 
            
            # 处理缺失视觉特征的关键逻辑：
            # 如果视觉特征存在 (mask_expanded is True)，则使用其投影 v_proj_initial。
            # 如果视觉特征缺失 (mask_expanded is False)，则使用 self.vis_dummy。
            v_proj = torch.where(mask_expanded, v_proj_initial, self.vis_dummy.expand_as(v_proj_initial))
        else:
            # 如果没有提供 vis_mask，则假设所有视觉特征都存在。
            v_proj = v_proj_initial

        if text_mask is not None:
            text_mask = text_mask.to(self.device)

        for layer in self.layers:
            # --- 1. 文本到视觉的注意力 (t_proj 作为 query, v_proj 作为 key/value) ---
            # 你的核心需求：如果 v_feat 缺失，t_proj 不应该关注它。
            
            # key_padding_mask: 在MultiheadAttention中, True 代表对应位置的key/value将被忽略。
            # 如果 vis_mask 为 False (视觉特征缺失), 那么 ~vis_mask 为 True (应该忽略)。
            vis_key_padding_mask = ~vis_mask if vis_mask is not None else None 
            if vis_key_padding_mask is not None:
                vis_key_padding_mask = vis_key_padding_mask.bool()

            t_attn, _ = layer['t2v_attn'](
                query=t_proj,
                key=v_proj, # 此处的 v_proj 对于缺失的视觉部分是 self.vis_dummy
                value=v_proj, # 同上
                key_padding_mask=vis_key_padding_mask # 这个掩码确保了 t_proj 会忽略掉 self.vis_dummy
            )
            t_proj = layer['norm1'](t_proj + layer['dropout'](t_attn))

            # --- 2. 视觉到文本的注意力 (v_proj 作为 query, t_proj 作为 key/value) ---
            # 目标：如果 v_feat 缺失，其代表者 vis_dummy 仍然可以去关注文本，
            # 从而形成一个基于上下文的“缺失视觉”的表征。
            text_key_padding_mask = ~text_mask if text_mask is not None else None
            if text_key_padding_mask is not None:
                text_key_padding_mask = text_key_padding_mask.bool()

            v_attn, _ = layer['v2t_attn'](
                query=v_proj, # 此处的 v_proj 对于缺失的视觉部分是 self.vis_dummy
                key=t_proj,
                value=t_proj,
                key_padding_mask=text_key_padding_mask
            )
            v_proj = layer['norm2'](v_proj + layer['dropout'](v_attn))

        fused_output = self.fusion_head(t_proj, v_proj)

        print(f"fused_output.shape: {fused_output.shape}")
        return fused_output