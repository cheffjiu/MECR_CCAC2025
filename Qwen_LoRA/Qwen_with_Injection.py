from transformers import PreTrainedModel, GenerationMixin
import torch
import torch.nn as nn
import torch.nn.functional as F

class QwenWithInjection(PreTrainedModel, GenerationMixin):
    def __init__(self, qwen_model, injection_module):
        super().__init__(qwen_model.config)
        self.backbone = qwen_model.model       # 原Qwen主干
        self.lm_head = qwen_model.lm_head
        self.injection_module = injection_module

        # 代理 generate 所需接口
        self.base_model = qwen_model  # PEFT 内部访问用

    def forward(self, input_ids, attention_mask=None, h_change=None, labels=None, **kwargs):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        last_hidden = outputs.hidden_states[-1]
        print("last_hidden:", last_hidden.shape)
        injected = self.injection_module(last_hidden, h_change)
        logits = self.lm_head(injected)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

    # === 关键接口 ===
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.base_model.set_input_embeddings(new_embeddings)

    def tie_weights(self):
        return self.base_model.tie_weights()

    def _init_weights(self, module):
        return self.base_model._init_weights(module)
