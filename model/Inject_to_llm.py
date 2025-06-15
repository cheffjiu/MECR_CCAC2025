import torch
import torch.nn as nn
import torch.nn.functional as F

class InjectionModule(nn.Module):
    """
    GNN 特征投影模块 (Projector) - 带有门控机制。
    负责将 GNN 的全局输出特征 h_change 投影为一系列“伪词元”嵌入，
    用于在输入层与文本嵌入进行拼接 (前注入)。
    门控机制确保了在训练初期，注入的信号极其微弱，以保证LLM的稳定性。
    """

    def __init__(self, d_gnn: int, d_model: int, num_gnn_tokens: int = 8, **kwargs):
        """
        初始化投影模块。

        Args:
            d_gnn (int): GNN 输出特征的维度 
            d_model (int): LLM 隐藏层的维度 
            num_gnn_tokens (int, optional): 希望生成的 GNN 伪词元的数量 (k)。默认为 8。
            **kwargs: 接收并忽略任何不使用的参数。
        """
        super().__init__()

        self.num_gnn_tokens = num_gnn_tokens

        self.projector = nn.Sequential(
            nn.Linear(d_gnn, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * self.num_gnn_tokens),
        )
        # 创建一个可训练的标量参数，作为门控。
        # 将其初始化为一个非常小的正数，这是确保训练初期稳定性的关键。
        self.layer_norm = nn.LayerNorm(d_model)
        # 您可以尝试 1e-3, 1e-4, 甚至 0.01，但 1e-4 是一个非常安全的选择。
        # ================================================================
        
        self._init_weights()

    def _init_weights(self):
        """初始化投影器权重"""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, h_change: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播，将 h_change 转换为经过门控的伪词元嵌入。

        Args:
            h_change (torch.Tensor): GNN 的输出特征，形状为 (B, d_gnn)。

        Returns:
            torch.Tensor: GNN 伪词元嵌入，形状为 (B, k, d_model)。
        """
        if h_change is None:
            raise ValueError("`h_change` cannot be None in the projector module.")

        if h_change.dim() != 2:
            raise ValueError(
                f"h_change should be a 2D tensor (B, d_gnn), but got shape {h_change.shape}"
            )

        # 1. 将 h_change 通过投影器
        projected_feats = self.projector(h_change)

        # 2. 将投影后的特征重塑为序列形式
        batch_size = h_change.shape[0]
        # 从projector的输出推断d_model，确保与reshape一致
        d_model = projected_feats.shape[1] // self.num_gnn_tokens
        gnn_embeds = projected_feats.view(batch_size, self.num_gnn_tokens, d_model)
        norm_gnn_embeds = self.layer_norm(gnn_embeds)
        return norm_gnn_embeds
       