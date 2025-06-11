import torch
import torch.nn as nn
import torch.nn.functional as F


class InjectionModule(nn.Module):
    """
    【重构后】GNN 特征投影模块 (Projector)。
    负责将 GNN 的全局输出特征 h_change 投影为一系列“伪词元”嵌入，
    用于在输入层与文本嵌入进行拼接 (前注入)。
    """

    def __init__(self, d_gnn: int, d_model: int, num_gnn_tokens: int = 8, **kwargs):
        """
        初始化投影模块。

        Args:
            d_gnn (int): GNN 输出特征的维度 (例如 1024)。
            d_model (int): LLM 隐藏层的维度 (即词嵌入维度，例如 4096)。
            num_gnn_tokens (int, optional): 希望生成的 GNN 伪词元的数量 (k)。默认为 8。
            **kwargs: 接收并忽略旧的、不再使用的参数 (如 n_heads, dropout)，以保持向后兼容。
        """
        super().__init__()

        self.num_gnn_tokens = num_gnn_tokens

        # 定义一个简单的多层感知机 (MLP) 作为投影器
        # 它的任务是将 [B, d_gnn] 映射到 [B, k * d_model]
        self.projector = nn.Sequential(
            nn.Linear(d_gnn, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * self.num_gnn_tokens),
        )

        # (可选) 可以保留权重初始化，这对于新模块是个好习惯
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
        执行前向传播，将 h_change 转换为伪词元嵌入。

        Args:
            h_change (torch.Tensor): GNN 的输出特征，形状为 (B, d_gnn)。
                                      注意：不再接收 h_llm。

        Returns:
            torch.Tensor: GNN 伪词元嵌入，形状为 (B, k, d_model)。
        """
        if h_change is None:
            # 在新架构中，我们总是期望 h_change 存在。如果不存在，可以返回空或报错。
            # 这里我们假设它总是存在。
            raise ValueError("`h_change` cannot be None in the new projector module.")

        if h_change.dim() != 2:
            raise ValueError(
                f"h_change should be a 2D tensor (B, d_gnn), but got shape {h_change.shape}"
            )

        # 1. 将 h_change 通过投影器
        # 输入: [B, d_gnn] -> 输出: [B, k * d_model]
        projected_feats = self.projector(h_change)

        # 2. 将投影后的特征重塑为序列形式
        # 获取批大小 B 和 LLM 模型维度 d_model
        batch_size = h_change.shape[0]
        d_model = projected_feats.shape[1] // self.num_gnn_tokens

        # 重塑: [B, k * d_model] -> [B, k, d_model]
        gnn_embeds = projected_feats.view(batch_size, self.num_gnn_tokens, d_model)

        return gnn_embeds
