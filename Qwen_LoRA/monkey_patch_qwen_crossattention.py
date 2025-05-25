import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir=os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, 'Qwen_LoRA'))



from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as QwenBlock
from Qwen_LoRA.cross_attention import HChangeCrossAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from peft.tuners.lora import LoraLayer

def replace_self_attn_with_lora(module, lora_config):
    def new_forward(self, hidden_states, *args, h_change=None, **kwargs):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.attn(hidden_states, *args, **kwargs)

        # 注入 CrossAttention 模块
        hidden_states = self.cross_attn(hidden_states, h_change)

        hidden_states = residual + hidden_states

        ffn_residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ffn_residual + hidden_states

        return hidden_states

    # 注入 cross_attn 实例
    def new_init(self, config):
        super(QwenBlock, self).__init__()
        self.attn = self.attn
        self.mlp = self.mlp
        self.ln_1 = self.ln_1
        self.ln_2 = self.ln_2
        self.cross_attn = HChangeCrossAttention(hidden_dim=config.hidden_size)

    QwenBlock.__init__ = new_init
    QwenBlock.forward = new_forward

def patch_qwen3_block_with_crossattention():
    # 使用正确的解码器层名称
    original_forward = Qwen2DecoderLayer.forward
    
    def new_forward(self, *args, **kwargs):
        # 跨模态注意力逻辑
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.attn(hidden_states, *args, **kwargs)

        # 注入 CrossAttention 模块
        hidden_states = self.cross_attn(hidden_states, h_change)

        hidden_states = residual + hidden_states

        ffn_residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ffn_residual + hidden_states

        return hidden_states

    # 注入 cross_attn 实例
    def new_init(self, config):
        super(QwenBlock, self).__init__()
        self.attn = self.attn
        self.mlp = self.mlp
        self.ln_1 = self.ln_1
        self.ln_2 = self.ln_2
        self.cross_attn = HChangeCrossAttention(hidden_dim=config.hidden_size)

    QwenBlock.__init__ = new_init
    QwenBlock.forward = new_forward
