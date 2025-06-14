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
        [重构后] 包装 Qwen 模型，实现 GNN 特征的“前注入”。
        """
        super().__init__(qwen_model.config)
        self.config = qwen_model.config
        self.cfg = cfg_model

        # 1. GNN 模块，保持不变
        self.multimodal_emotion_gnn = MultimodalEmotionGNN(
            # 融合参数初始化
            t_feat_dim=self.cfg.fusion_d_t,
            v_feat_dim=self.cfg.fusion_d_v,
            d_fusion=self.cfg.fusion_d_fusion,
            fusion_heads=self.cfg.fusion_num_heads,
            fusion_layers=self.cfg.fusion_num_layers,
            fusion_dropout=self.cfg.fusion_dropout,
            # GNN 模块参数初始化
            gnn_in_dim=self.cfg.gnn_in_dim,
            gnn_hidden_dim=self.cfg.gnn_hidden_dim,
            gnn_out_dim=self.cfg.gnn_out_dim,
            gnn_heads=self.cfg.gnn_num_heads,
            gnn_dropout=self.cfg.gnn_dropout,
        )

        # 2. LLM 主干和头部，保持不变
        self.backbone = qwen_model.model
        self.lm_head = qwen_model.lm_head

        # 3. 使用修改后的 InjectionModule 作为 GNN 投影器
        self.injection_module = InjectionModule(
            d_gnn=self.cfg.injection_in_dim,  # GNN 输出维度
            d_model=self.config.hidden_size,  # LLM 的隐藏/词嵌入维度
            num_gnn_tokens=self.cfg.injection_num_gnn_tokens,  # GNN 伪词元数量 k
        )
        # 将 k 保存为实例属性，方便 forward 中使用
        self.num_gnn_tokens = self.cfg.injection_num_gnn_tokens

        # 4. 其他初始化，保持不变
        self.generation_config = qwen_model.generation_config
        self.main_input_name = "input_ids"
        self._freeze_backbone()

    def _freeze_backbone(self):
        """冻结主干参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Qwen 主干参数已冻结")

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.backbone.set_input_embeddings(new_embeddings)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # 这个labels是只屏蔽了文本prompt的
        batched_graph: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # 只有在第一步（无KV缓存）时才注入GNN
        if past_key_values is None and batched_graph is not None:
            h_change, _ = self.multimodal_emotion_gnn(batched_graph)
            gnn_embeds_orig = self.injection_module(h_change)
            
            # 【终极的、处理Beam Search的核心修复】
            # 获取文本嵌入和GNN嵌入的批次大小
            bs_text_embeds = inputs_embeds.shape[0]
            bs_gnn_embeds = gnn_embeds_orig.shape[0]

            if bs_text_embeds != bs_gnn_embeds:
                # 如果不匹配，说明 .generate() 进行了扩展 (bs_text_embeds = bs_gnn_embeds * num_beams)
                num_beams = bs_text_embeds // bs_gnn_embeds
                # 我们需要手动将GNN嵌入扩展到相同的批次大小
                gnn_embeds = gnn_embeds_orig.unsqueeze(1).expand(
                    -1, num_beams, -1, -1
                ).reshape(bs_text_embeds, self.num_gnn_tokens, -1)
            else:
                # 如果批次大小相同（训练或num_beams=1），直接使用
                gnn_embeds = gnn_embeds_orig
            
            # 现在，gnn_embeds 和 inputs_embeds 的批次大小保证是一致的
            inputs_embeds = torch.cat([gnn_embeds, inputs_embeds], dim=1)
           
            
            if attention_mask is not None:
                gnn_attention_mask = torch.ones(
                    gnn_embeds.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([gnn_attention_mask, attention_mask], dim=1)
        
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            #  在模型内部完成 GNN 伪词元的 label 屏蔽
            
            # 1. 创建一个用于屏蔽GNN伪词元的-100张量
            batch_size = inputs_embeds.shape[0]
            padding_for_gnn_labels = torch.full(
                (batch_size, self.num_gnn_tokens), -100, dtype=torch.long, device=labels.device
            )
            
            # 2. 将 -100 填充与 Trainer 传入的、已屏蔽了prompt的labels拼接
            #    这样就得到了与 logits 完全对齐的、最终的 aligned_labels
            aligned_labels = torch.cat([padding_for_gnn_labels, labels], dim=1)
            
            # 3. 使用对齐后的 a_labels 计算损失
            loss_fct = nn.CrossEntropyLoss()
            
            # 为了安全，确保对齐后的标签长度不超过logits的序列长度
            logits_len = logits.shape[1]
            aligned_labels = aligned_labels[:, :logits_len]

            loss = loss_fct(logits.view(-1, self.config.vocab_size), aligned_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        
        # 标准的KV缓存处理
        if past_key_values:
            input_ids = input_ids[:, -1:]

        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}
        
        # 【LLaVA范式 - D】: 智能传递 batched_graph
        # 只有在第一步（无KV缓存）时，我们才需要 batched_graph
        if not past_key_values:
            model_inputs['batched_graph'] = kwargs.get('batched_graph', None)

        # 传递其他所有必要的kwargs，尤其是 attention_mask
        model_inputs.update(kwargs)
        
        return model_inputs
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
        
    def tie_weights(self):
        self.backbone.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        return self.backbone.resize_token_embeddings(new_num_tokens)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )