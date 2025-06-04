from dataclasses import dataclass
from typing import Optional
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path)))


@dataclass
class config_feature_fusion_model:
    d_t: int = 768  # 文本特征维度
    d_v: int = 512  # 视频特征维度
    d_fusion: int = 256  # 融合特征维度
    num_heads: int = 4  # 多头注意力头数
    num_layers: int = 2  # 交叉注意力层数
    dropout: float = 0.1  # 丢弃率


@dataclass
class config_emotion_graph_model:
    gnn_in_dim: int = 257  # GNN输入维度=融合特征维度+1(位置编码)
    gnn_hidden_dim: int = 256  # GNN隐藏层维度
    gnn_out_dim: int = 256  # GNN输出维度
    num_heads: int = 4  # GNN多头注意力头数
    dropout: float = 0.1  # 丢弃率


@dataclass
class config_injection_module:
    d_gnn: int = 1024  # GNN输出维度
    d_model: int = 1024  # LLM模型输出维度
    n_heads: int = 4  # 多头注意力头数
    dropout: float = 0.1  # 丢弃率


@dataclass
class config_qwen_llm:
    llm_name: str = "Qwen/Qwen3-0.6B"  # 模型名称或路径


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
    batch_size: int = 16  # 批大小
    num_workers: int = 0  # 工作进程数


@dataclass
class config_train:
    # ===配置训练参数===#
    num_train_epochs: int = 1  # 训练轮数
    learning_rate: float = 1e-4  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    warmup_ratio: float = 0.05  # 权重衰减
    accumulation_steps: int = 1  # 梯度累积步数
    max_grad_norm: float = 1.0  # 梯度裁剪阈值
    # ===早停参数===#
    patience: int = 5  # 早停轮数
    min_delta: float = 0.001  # 早停最小变化
    # ===配置模型保存路径===#
    model_save_path: str = os.path.join(
        project_root, "checkpoints", "model.pth"
    )  # 模型保存路径


@dataclass
class config_lora:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


@dataclass
class config:
    cfg_feature_fusion_model: config_feature_fusion_model
    cfg_emotion_graph_model: config_emotion_graph_model
    cfg_injection_module: config_injection_module
    cfg_qwen_llm: config_qwen_llm
    cfg_dataset_dataloader: config_dataset_dataloader
    cfg_train: config_train
    cfg_lora: config_lora
