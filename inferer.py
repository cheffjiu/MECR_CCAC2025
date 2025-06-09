import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # 关闭tokenizer的并行化，避免与dataloader的多线程冲突，导致死锁
)


current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path)))
sys.path.extend(
    os.path.join(project_root, sub) for sub in ["", "model", "MECE_data", "utils"]
)
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from dataclasses import dataclass
from accelerate import Accelerator


# Import PEFT modules
from peft import LoraConfig, get_peft_model, TaskType

# 导入自定义模块
from model.Qwen_with_Injection import QwenWithInjection
from MECE_data.mecr_dataset import MECRDataset
from MECE_data.collate_to_graph_batch import CustomCollate
from MECE_data.build_emotion_graph import build_emotion_graph

from utils.rationale_evaluate import RationaleEvaluator

class Inferer:
    def __init__(self, cfg: dataclass):
        self.config = cfg
        self.device = self.accelerator.device

        # 2. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.cfg_dataset_dataloader.tokenizer_name, trust_remote_code=True
        )

        # 3. 初始化模型
        qwen_base_model = AutoModelForCausalLM.from_pretrained(
            self.config.cfg_model.llm_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        self.model = QwenWithInjection(
            qwen_base_model,
            self.config.cfg_model,
        )

        # 4. 数据集和数据加载器
        self.train_dataset = MECRDataset(
            json_path=self.config.cfg_dataset_dataloader.json_path_demo,
            feature_root=self.config.cfg_dataset_dataloader.feature_root_demo,
            mode="test",
            tokenizer=self.config.cfg_dataset_dataloader.tokenizer_name,
            bert_model=self.config.cfg_dataset_dataloader.bert_name,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.cfg_dataset_dataloader.batch_size,
            shuffle=True,
            collate_fn=CustomCollate(build_emotion_graph),
            num_workers=self.config.cfg_dataset_dataloader.num_workers,
            pin_memory=True,
        )