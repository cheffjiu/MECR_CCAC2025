# test_GNN.py

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

# -------------------- 路径配置和导入 --------------------
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "feature_fusion"))

from feature_fusion.fusion_feature_dataset import FusionFeatureDataset
from feature_fusion.fusion_model import CrossModalAttention
from build_emotion_graph import build_emotion_graph
from GAT_modal import EmotionGraphEncoder
from collate_fn import collate_to_graph_batch
from torch_geometric.data import Batch

# -------------------- 配置 --------------------
FEATURE_ROOT = os.path.join(project_root, "data/feature/demo")
JSON_PATH = os.path.join(project_root, "data/processed/demo_cleaned/demo_cleaned.json")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 1
D_TEXT = 768
D_VIDEO = 512
D_MODEL = 256
GRAPH_HIDDEN_DIM = 256
GRAPH_OUT_DIM = 256

# -------------------- LLM BERT 模型和 Tokenizer 加载 --------------------
print("正在加载 BERT 模型和 Tokenizer (bert-base-chinese) 用于 FAISS 查询...")
try:
    llm_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    llm_bert_model = AutoModel.from_pretrained("bert-base-chinese")
    llm_bert_model.to(DEVICE)
    llm_bert_model.eval()
    for param in llm_bert_model.parameters():
        param.requires_grad = False
    print("BERT 模型和 Tokenizer 加载成功。")
    with_prompt_enabled = True
except Exception as e:
    print(f"警告: 无法加载 BERT 模型或 Tokenizer。将跳过 LLM 提示生成。错误: {e}")
    llm_tokenizer = None
    llm_bert_model = None
    with_prompt_enabled = False

# --- 1. 数据集和数据加载器 ---
print(f"正在从 {JSON_PATH} 加载数据，并从 {FEATURE_ROOT} 加载特征...")
dataset = FusionFeatureDataset(
    feature_root=FEATURE_ROOT,
    json_path=JSON_PATH,
    mode="train",
    tokenizer=llm_tokenizer,
    bert_model=llm_bert_model,
    device=DEVICE
)
print(f"数据集加载完成。样本数量: {len(dataset)}")

# --- 2. 初始化模型 ---
print(f"正在设备 {DEVICE} 上初始化模型...")
fusion_model = CrossModalAttention(
    d_t=D_TEXT, d_v=D_VIDEO, d_model=D_MODEL, device=DEVICE
)
graph_encoder = EmotionGraphEncoder(
    in_dim=D_MODEL, hidden_dim=GRAPH_HIDDEN_DIM, out_dim=GRAPH_OUT_DIM, device=DEVICE
)

# --- 3. 准备数据整理函数 ---
graph_builder_fn = build_emotion_graph

collate_batch_fn = lambda b: collate_to_graph_batch(
    b,
    fusion_model,
    graph_builder_fn,
    with_prompt=with_prompt_enabled,
)

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch_fn
)
print("DataLoader 初始化完成。")

# --- 4. 管道的前向传播 ---
print("正在开始管道的前向传播...")
for i, (batch_graph_data, prompt_texts, label_texts) in enumerate(dataloader):
    print(f"\n正在处理批次 {i+1}/{len(dataloader)}")
    print(f"  批次图数据: {batch_graph_data}")

    batch_graph_data = batch_graph_data.to(DEVICE)

    h_change, h_all = graph_encoder(batch_graph_data)

    print(f"  输出 h_change 形状: {h_change.shape}")
    print(f"  输出 h_all (所有节点) 形状: {h_all.shape}")

    if with_prompt_enabled and prompt_texts and label_texts:
        print(f"  LLM 提示:\n{prompt_texts[0]}")
        # === 唯一修改点：这里将 "LLM 标签" 改为 "Ground-true label" ===
        print(f"  Ground-true label:\n{label_texts[0]}")
        # =======================================================
    else:
        print(
            "  批次中未找到 LLM 提示或标签数据（可能 with_prompt 为 False 或数据缺失）。"
        )

print("\n管道执行完成！")