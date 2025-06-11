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
        attention_mask: Optional[torch.Tensor] = None,
        batched_graph: Optional[Any] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # 判断是否是推理的第一步 (没有KV缓存) 或 训练 (没有KV缓存)
        is_first_step_or_training = past_key_values is None

        if is_first_step_or_training:
            # 如果是第一步或训练，需要执行完整的“前注入”流程
            if inputs_embeds is not None:
                # 正常不应发生，如果外部传入了 embeds，优先使用
                pass
            elif input_ids is not None:
                # 1. 获取 GNN 特征 h_change
                if batched_graph is None:
                    raise ValueError(
                        "`batched_graph` must be provided for the first step or training."
                    )
                h_change, _ = self.multimodal_emotion_gnn(batched_graph)

                # 2. 投影 h_change 为伪词元嵌入
                gnn_embeds = self.injection_module(h_change)

                # 3. 获取文本词嵌入
                text_embeds = self.get_input_embeddings()(input_ids)

                # 4. 拼接嵌入向量
                inputs_embeds = torch.cat([gnn_embeds, text_embeds], dim=1)

                # 5. 更新 attention_mask 以包含 GNN 伪词元
                if attention_mask is not None:
                    gnn_attention_mask = torch.ones(
                        gnn_embeds.size()[:-1],
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                    attention_mask = torch.cat(
                        [gnn_attention_mask, attention_mask], dim=1
                    )
            else:
                raise ValueError(
                    "You have to specify either `input_ids` or `inputs_embeds`."
                )

        # 将最终的嵌入向量或原始输入送入 LLM backbone
        # 在推理的后续步骤中，inputs_embeds 为 None，backbone 将使用 input_ids (单个新 token) 和 past_key_values
        backbone_outputs = self.backbone(
            input_ids=(
                None if is_first_step_or_training else input_ids
            ),  # 在第一步或训练时，input_ids 已被处理
            inputs_embeds=inputs_embeds if is_first_step_or_training else None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 计算 logits
        hidden_states = backbone_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            # 裁剪 logits 来匹配原始 labels 的长度
            # 丢弃 GNN 伪词元对应的 logits
            text_logits = logits[:, self.num_gnn_tokens :, :].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (logits,) + backbone_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,  # generate 需要完整的 logits
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )

    # =========== 生成相关 ===========
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        为生成步骤准备输入。这是适配 .generate() 的关键。
        """
        # 这是 Hugging Face 的标准模板

        # 当使用 KV 缓存时，我们只需要最后一个 token 作为新的 input_ids
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 将 `batched_graph` 和其他重要的 kwargs 打包，
        # 以便 .generate() 能将它们全部传递给 forward 方法。
        # `attention_mask` 和 `use_cache` 也会在 kwargs 中。
        kwargs["past_key_values"] = past_key_values
        return {"input_ids": input_ids, **kwargs}

    def _reorder_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        beam_idx: torch.LongTensor,
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        为 beam search 正确地重新排序 KV 缓存。
        这个实现是标准的，无需改动。
        """
        return tuple(
            tuple(layer_past.index_select(0, beam_idx) for layer_past in layer)
            for layer in past_key_values
        )

    def can_generate(self) -> bool:
        return True

    # ========== LoRA / PEFT 相关方法保持不变 ==========

    def tie_weights(self):
        self.backbone.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        return self.backbone.resize_token_embeddings(new_num_tokens)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings
