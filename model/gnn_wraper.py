import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# 导入您已经写好的GNN主模型
from multimodal_emotion_gnn import MultimodalEmotionGNN


class ContrastiveWrapper(nn.Module):
    """
    一个包装器（Wrapper）模型，专门用于“阶段零”的对比学习预训练。
    它内部包含 GNN 编码器和一个文本编码器。
    """

    def __init__(
        self,
        gnn_model: MultimodalEmotionGNN,
        text_encoder_name: str,
        projection_dim: int = 256,
    ):
        super().__init__()
        # 1. 核心组件：您提供的GNN模型和新指定的文本编码器
        self.graph_encoder = gnn_model
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

        # 2. 动态获取图和文本的输出维度
        # 从GNN模块的最终MLP层动态获取图特征的输出维度
        gnn_output_dim = self.graph_encoder.gnn_module.mlp[-1].out_features
        text_output_dim = self.text_encoder.config.hidden_size

        # 3. 两个独立的投影头，将不同模态的输出映射到同一个维度的共享空间
        self.graph_proj = nn.Linear(gnn_output_dim, projection_dim)
        self.text_proj = nn.Linear(text_output_dim, projection_dim)

        # 4. 可学习的温度参数，用于调整softmax的锐度，这是CLIP中的标准做法
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # 初始化为 log(1/0.07)

    def forward(self, batched_graph, text_list: list):
        # 将输入数据移动到模型所在的设备
        device = self.graph_proj.weight.device
        batched_graph = batched_graph.to(device)

        # --- 编码图数据 ---
        # 调用您已有的GNN模型，只取我们需要的h_change
        graph_features, _ = self.graph_encoder(batched_graph)
        graph_embeds = self.graph_proj(graph_features)

        # --- 编码文本数据 ---
        # 使用内置的tokenizer处理文本
        tokenized_texts = self.text_tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=256,  # rationale文本不长，可以适当限制
            return_tensors="pt",
        ).to(device)

        text_outputs = self.text_encoder(**tokenized_texts)
        # 使用[CLS] token的输出作为整个句子的嵌入表示
        text_cls_embeds = text_outputs.last_hidden_state[:, 0]
        text_embeds = self.text_proj(text_cls_embeds)

        # --- L2归一化 ---
        # 将嵌入向量归一化为单位向量，这样它们的点积就是余弦相似度
        graph_embeds = F.normalize(graph_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        """标准的InfoNCE对比损失函数"""
        logits_per_graph = self.logit_scale * graph_embeds @ text_embeds.t()
        logits_per_text = logits_per_graph.t()

        batch_size = len(graph_embeds)
        labels = torch.arange(batch_size, device=graph_embeds.device)

        loss = (
            F.cross_entropy(logits_per_graph, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return loss
