from transformers import PreTrainedModel, GenerationMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union


class QwenWithInjection(PreTrainedModel, GenerationMixin):
    def __init__(self, qwen_model, injection_module, freeze_backbone: bool = False):
        """
        优化后的 Qwen 模型注入 GNN 特征

        主要优化点:
        1. 支持梯度控制 (冻结主干/仅训练注入模块)
        2. 改进的生成接口支持
        3. 内存高效设计
        4. 完整的类型注解
        5. 错误处理和调试支持

        Args:
            qwen_model: 基础 Qwen 模型
            injection_module: 特征注入模块
            freeze_backbone: 是否冻结 Qwen 主干参数
        """
        super().__init__(qwen_model.config)
        self.config = qwen_model.config

        # 模型组件
        self.backbone = qwen_model.model
        self.lm_head = qwen_model.lm_head
        self.injection_module = injection_module

        # 梯度控制
        if freeze_backbone:
            self._freeze_backbone()

        # 代理生成所需接口
        self.base_model = qwen_model
        self.generation_config = qwen_model.generation_config
        self.main_input_name = "input_ids"

    def _freeze_backbone(self):
        """冻结 Qwen 主干参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Qwen 主干参数已冻结")

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        h_change: Optional[torch.Tensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        """
        前向传播，支持训练和推理

        设计原则:
        1. 兼容标准 Hugging Face 模型接口
        2. 支持 past_key_values 用于高效生成
        3. 内存高效，避免不必要计算
        """
        # 设置默认输出行为
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # 运行基础模型
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # 获取最后一层隐藏状态
        if return_dict:
            last_hidden_state = backbone_outputs.last_hidden_state
        else:
            last_hidden_state = backbone_outputs[0]

        print(f"基础模型输出形状: {last_hidden_state.shape}")

        # 特征注入
        injected_hidden = self.injection_module(last_hidden_state, h_change)

        # 语言建模头
        logits = self.lm_head(injected_hidden)

        # 计算损失 (如果提供标签)
        loss = None
        if labels is not None:
            # 展平张量
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        # 返回结果
        if not return_dict:
            output = (logits,) + backbone_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": backbone_outputs.past_key_values,
            "hidden_states": backbone_outputs.hidden_states,
            "attentions": backbone_outputs.attentions,
            "injected_hidden": injected_hidden,
        }

    # ===== 关键生成接口 =====
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        h_change: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        准备生成输入，支持 h_change 参数
        """
        # 调用基础模型准备方法
        model_inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        # 添加 h_change 参数
        model_inputs["h_change"] = h_change

        print(
            f"生成输入准备: input_ids={input_ids.shape}, h_change={h_change.shape if h_change is not None else None}"
        )

        return model_inputs

    def get_input_embeddings(self) -> nn.Module:
        """获取输入嵌入层"""
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Module):
        """设置输入嵌入层"""
        self.backbone.set_input_embeddings(new_embeddings)

    def tie_weights(self):
        """绑定权重"""
        self.backbone.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """调整 token 嵌入大小"""
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def get_output_embeddings(self) -> nn.Module:
        """获取输出嵌入层"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        """设置输出嵌入层"""
        self.lm_head = new_embeddings

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """重新排序缓存以支持 beam search"""
        return self.base_model._reorder_cache(past_key_values, beam_idx)

    def can_generate(self) -> bool:
        """指示模型是否支持生成"""
        return True
