# Qwen_LoRA/modify_qwen_for_final_injection.py (Revised for MLP LoRA + Final Layer Injection)
import torch
import torch.nn as nn
import sys
import os
from typing import List, Optional
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from Injection_Module import InjectionModule 
# 导入 Qwen2 模型的核心组件,因为Qwen3模型是以Qwen2为基础的，所以我们可以直接导入Qwen2的模型组件
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM, Qwen2DecoderLayer, Qwen2MLP


# --- 1. 定义注入模块的添加函数 ---
def setup_final_qwen_injection(model: Qwen2ForCausalLM, d_gnn_output_dim: int, n_heads: int):
    """
    在 Qwen2ForCausalLM 模型的 lm_head 之前添加一个 InjectionModule。
    这个 InjectionModule 将在 LLM 的最终隐藏状态和 lm_head/MLP 之间进行融合。

    Args:
        model (Qwen2ForCausalLM): Qwen2 模型实例。
        d_gnn_output_dim (int): GNN 输出 h_change 的维度。
        n_heads (int): InjectionModule 内部交叉注意力的头数。
    """
    # 确保 InjectionModule 能够访问到 LLM 的 hidden_size (d_model)
    d_model = model.config.hidden_size 

    if not hasattr(model, 'final_injection_module'):
        model.final_injection_module = InjectionModule(
            d_gnn=d_gnn_output_dim,
            d_model=d_model,
            n_heads=n_heads
        ).to(model.device) # 确保 InjectionModule 在正确的设备上
        print("Added final InjectionModule to Qwen2ForCausalLM before lm_head/MLP.")
    else:
        print("Final InjectionModule already exists in Qwen2ForCausalLM.")

# --- 2. 修改 Qwen2ForCausalLM 的 forward 方法 ---
# 保存原始的 Qwen2ForCausalLM forward 方法
_original_qwen2_for_causal_lm_forward = Qwen2ForCausalLM.forward

def patched_qwen2_for_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    h_change: Optional[torch.Tensor] = None, # 传递 GNN 的 h_change
):
    """
    Modified Qwen2ForCausalLM forward method to inject h_change
    before the final lm_head (and thus before it conceptually hits the MLP if it were part of a block).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 调用原始 Qwen2Model 的 forward 方法以获取最后一层隐藏状态
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=True, # 确保获取到所有隐藏状态，特别是最后一层
        return_dict=True,
    )

    # h_llm 是 LLM 的最后一层隐藏状态
    # outputs.last_hidden_state 的形状是 (batch_size, sequence_length, hidden_size)
    h_llm = outputs.last_hidden_state

    # --- 注入点：在 Qwen2 的 MLP/lm_head 之前，将 h_llm 和 h_change 融合 ---
    if hasattr(self, 'final_injection_module') and self.final_injection_module is not None:
        # 融合 h_llm 和 h_change，得到 m
        m = self.final_injection_module(h_llm, h_change)
    else:
        # 如果没有注入模块，则 m 就是原始的 h_llm
        m = h_llm

    # 将融合后的 `m` 传递给 lm_head
    # PEFT 库在应用 LoRA 时，会自动修改目标模块的 `forward` 方法。
    # 如果 `target_modules` 包含 MLP 层的投影（如 `gate_proj`, `up_proj`, `down_proj`），
    # 那么当 `m` 经过 Qwen2 的 `mlp` 层时，LoRA 就会起作用。
    # Qwen2ForCausalLM 的 lm_head 通常直接从 last_hidden_state 预测 logits。
    # 这里的 `m` 已经是融合后的状态，它将作为 `lm_head` 的输入。
    logits = self.lm_head(m).float() # Qwen2 的 lm_head

    loss = None
    if labels is not None:
        # 将 logits 和 labels 移动到同一设备进行计算
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism for loss calculation (if applicable)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def patch_qwen_for_final_injection():
    """
    将 Qwen2ForCausalLM 的 forward 方法打补丁，以在最终 lm_head 之前注入 h_change。
    """
    Qwen2ForCausalLM.forward = patched_qwen2_for_causal_lm_forward
    print("Qwen2ForCausalLM.forward has been patched for final hidden state injection.")

# --- 3. 参数冻结与训练控制 (适应新注入位置) ---
def freeze_qwen_except_lora_and_final_proj(model: nn.Module):
    """
    冻结 Qwen2 模型的参数，只解冻 LoRA 参数和最终 InjectionModule 的投影层参数。

    Args:
        model (nn.Module): Qwen2 模型实例 (例如 Qwen2ForCausalLM)。
    """
    # 首先冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    # PEFT 库在调用 get_peft_model 后，会自动将 LoRA 适配器中的 A 和 B 矩阵设置为 requires_grad=True
    # 我们只需要确保最终 InjectionModule 自身的非 LoRA 参数被解冻

    # 解冻最终 InjectionModule 的参数（增加交叉注意力层参数）
    if hasattr(model, 'final_injection_module') and model.final_injection_module is not None:
        print(f"Unfreezing parameters for final_injection_module.")
        for name, param in model.final_injection_module.named_parameters():
            # 确保交叉注意力层参数也被解冻
            if 'cross_attention' in name or 'fusion_gate' in name:
                param.requires_grad = True
            param.requires_grad = True  # 确保所有参数都被解冻
    
    # 特别解冻交叉注意力层的输出投影层
    if hasattr(model.final_injection_module, 'cross_attention'):
        for name, param in model.final_injection_module.cross_attention.named_parameters():
            param.requires_grad = True
    
    print("Qwen model parameters frozen, LoRA and final InjectionModule parameters unfrozen.")