from dataclasses import dataclass
from typing import Optional
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path)))


@dataclass
class config_model:
    # ===融合模块参数配置====#
    fusion_d_t: int = 768  # 文本特征维度
    fusion_d_v: int = 512  # 视频特征维度
    fusion_d_fusion: int = 256  # 融合特征维度
    fusion_num_heads: int = 4  # 多头注意力头数
    fusion_num_layers: int = 2  # 交叉注意力层数
    fusion_dropout: float = 0.1  # 丢弃率
    # ===GNN参数配置====#
    gnn_in_dim: int = 256  # GNN输入维度
    gnn_hidden_dim: int = 256  # GNN隐藏层维度
    gnn_out_dim: int = 256  # GNN输出维度
    gnn_num_heads: int = 4  # GNN多头注意力头数
    gnn_dropout: float = 0.1  # 丢弃率
    # ===LLM注入模块参数配置====#
    injection_in_dim: int = 1024  # GNN输出维度
    injection_out_dim: int = 1024  # LLM模型输出维度
    injection_num_gnn_tokens: int = 4  # 生成的伪词元数量
    # ===LLM参数配置====#
    llm_name: str = "Qwen/Qwen3-0.6B"  # 模型名称或路径
    llm_tokenizer_name: str = "Qwen/Qwen3-0.6B"


@dataclass
class config_dataset_dataloader:
    # ===FAISS检索时使用的分词器和BERT模型===#
    tokenizer_name: str = "bert-base-chinese"  # 分词器名称或路径
    bert_name: str = "bert-base-chinese"  # BERT模型名称或路径
    # ===配置数据集路径和对应特征路径===#
    json_path_demo: str = os.path.join(
        project_root, "data/processed/demo_cleaned/demo_cleaned.json"
    )
    feature_root_demo: str = os.path.join(project_root, "data/feature/demo")
    json_path_train: str = os.path.join(
        project_root, "data/processed/train_cleaned/train_cleaned.json"
    )
    feature_root_train: str = os.path.join(project_root, "data/feature/train")
    json_path_val: str = os.path.join(
        project_root, "data/processed/val_cleaned/val_cleaned.json"
    )
    feature_root_val: str = os.path.join(project_root, "data/feature/val")
    # ===配置dataloader参数===#
    batch_size: int = 1 # 批大小
    num_workers: int = 4  # 工作进程数


@dataclass
class config_train:
    # ===配置训练参数===#
    warmup_ratio: float = 0.1  # 预热率
    accumulation_steps: int = 2  # 梯度累积步数
    max_grad_norm: float = 1.0  # 梯度裁剪阈#
    #===第一阶段训练参数===#
    stage1_epochs :int =10 #训练轮数
    stage1_learning_rate:float =1e-5 #学习率
    stage1_weight_decay: float = 1e-4 # 权重衰减
    #===第二阶段训练参数===#
    stage2_epochs :int =10 #训练轮数
    stage2_lr_gnn:float = 1e-5 #GNN+融合模块学习率
    stage2_lr_lora: float = 1e-6 #lora学习率
    num_train_epochs: int = 10 # 训练轮数
    weight_decay: float = 1e-4  # 权重衰减

    # === 生成参数 ===
    # 针对情感回应生成，我们通常希望回应既有创造性又不过于离谱，避免重复
    max_new_tokens: int = 128  # 根据情感回应的典型长度调整，例如 80-120 词
    num_beams: int = 1  # 采样模式下通常设为1，不使用束搜索
    do_sample: bool = True  # 启用采样，增加回应多样性
    temperature: float = 0.8  # 略低于1，确保一定随机性但不过于发散
    top_p: float = 0.9  # 常用且效果好的 Top-P 值，在保证质量的同时增加多样性
    top_k: int = 50  # 与 top_p 配合使用时，通常设为0
    repetition_penalty: float = 1.1  # 适度惩罚重复，防止回应过于机械或陷入循环
    # ===早停参数===#
    start_eval_epoch: int = 1  # 开始验证epoch
    patience: int = 8  # 早停轮数
    min_delta: float = 0.001  # 早停最小变化
    # ===配置模型保存===#
    model_save_path: str = os.path.join(
        project_root, "checkpoints", "model.pth"
    )  # 模型保存路径


@dataclass
class config_lora:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


@dataclass
class config:
    cfg_model: config_model
    cfg_dataset_dataloader: config_dataset_dataloader
    cfg_train: config_train
    cfg_lora: config_lora
