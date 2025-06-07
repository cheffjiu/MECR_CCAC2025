from transformers import PreTrainedModel, GenerationMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from multimodal_emotion_gnn import MultimodalEmotionGNN
from Inject_to_llm import InjectionModule


class QwenWithInjection(PreTrainedModel, GenerationMixin):
    def __init__(self, qwen_model, cfg_model):
        """
        包装 Qwen 模型，实现 GNN 特征注入。

        Args:
            qwen_model: 原始 Qwen 模型
            injection_module: 注入模块（接收 LLM 隐状态 + h_change）
        """
        super().__init__(qwen_model.config)
        self.config = qwen_model.config
        self.cfg = cfg_model
        self.multimodal_emotion_gnn = MultimodalEmotionGNN(
            #=====融合模块参数初始化=====#
            t_feat_dim=self.cfg.fusion_d_t,
            v_feat_dim=self.cfg.fusion_d_v,
            d_fusion=self.cfg.fusion_d_fusion,
            fusion_heads=self.cfg.fusion_num_heads,
            fusion_layers=self.cfg.fusion_num_layers,
            fusion_dropout=self.cfg.fusion_dropout,
            #=====GNN参数初始化=====#
            gnn_in_dim=self.cfg.gnn_in_dim,
            gnn_hidden_dim=self.cfg.gnn_hidden_dim,
            gnn_out_dim=self.cfg.gnn_out_dim,
            gnn_heads=self.cfg.gnn_num_heads,
            gnn_dropout=self.cfg.gnn_dropout,
        )

        self.backbone = qwen_model.model
        self.lm_head = qwen_model.lm_head
        self.injection_module = InjectionModule(
            d_gnn=self.cfg.injection_in_dim,
            d_model=self.cfg.injection_out_dim,
            n_heads=self.cfg.injection_num_heads,
            dropout=self.cfg.injection_dropout,
        )


        self.generation_config = qwen_model.generation_config
        self.main_input_name = "input_ids"

        self._freeze_backbone()

    def _freeze_backbone(self):
        """冻结主干参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Qwen 主干参数已冻结")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        batched_graph: Optional[Any] = None,
        h_change: Optional[torch.Tensor] = None, #模型的generate()使用
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        """
        前向传播，训练 & 推理通用
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        #获取h_change
        h_change = None
        if h_change is None and batched_graph is not None:
            h_change, _ = self.multimodal_emotion_gnn(batched_graph)
            # print(f"GNN生成的h_change形状: {h_change.shape}, 类型: {h_change.dtype}")


        # 主干模型输出
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        last_hidden_state = (
            backbone_outputs.last_hidden_state if return_dict else backbone_outputs[0]
        )

        # print(f"基础模型输出形状: {last_hidden_state.shape}")

        # 注入 h_change 特征
        if h_change is not None:
            injected_hidden = self.injection_module(last_hidden_state, h_change)
        else:
            injected_hidden = last_hidden_state
        # print(f"注入后形状: {injected_hidden.shape}")

        logits = self.lm_head(injected_hidden)
        # print(f"语言模型logits输出形状: {logits.shape}")

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + backbone_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )
    

    # =========== 生成相关 ===========
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        batched_graph: Optional[Any] = None,
        h_change: Optional[torch.Tensor] = None,  # <<< 加入这个
        **kwargs,
    ) -> Dict[str, Any]:
        """
        为生成步骤准备输入，确保 h_change 在每一步都传入 forward。
        """

        # 如果 KV 缓存已存在，我们只需要将最后一个 token 作为输入。
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 如果当前 step 没传 h_change，手动从 batched_graph 生成（通常只会在第一步）
        if h_change is None and batched_graph is not None:
            h_change, _ = self.multimodal_emotion_gnn(batched_graph)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "batched_graph": batched_graph,  # optional，可用于 debug
            "h_change": h_change,  # <<< 持续传递
            **kwargs,
        }
    


    def _reorder_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        beam_idx: torch.LongTensor,
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(layer_past.index_select(0, beam_idx) for layer_past in layer)
            for layer in past_key_values
        )

    def can_generate(self) -> bool:
        return True

    # ========== 模型LoRA / PEFT 相关 ===========
    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.backbone.set_input_embeddings(new_embeddings)

    def tie_weights(self):
        self.backbone.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        return self.backbone.resize_token_embeddings(new_num_tokens)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings
