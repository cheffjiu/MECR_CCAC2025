import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 获取项目根目录
current_file_path = os.path.abspath(__file__)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))

# 加载 BERT 模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# 切换评估模式 + 冻结参数
model.eval()  # 切换到评估模式（关闭Dropout等）
for param in model.parameters():
    param.requires_grad = False  # 禁止参数更新

# 读取 JSON 文件
# 修改：将路径改为 demo.json 的位置
demo_json_path = os.path.join(workspace_root, "data/MECR_CCAC2025/train.json")
with open(demo_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 存储所有拼接后的对话文本和对应的完整元数据（对话+rationale+情感信息）
all_texts = []  # 用于生成嵌入的文本（对话内容）
all_metadata = []  # 存储对话+rationale+情感信息的完整元数据

# 遍历每个样本（修改：适配demo.json的对象结构并提取情感信息）
logging.info(f"开始处理样本数据，总样本数：{len(data)}")
for sample_id, sample in tqdm(data.items(), desc="样本处理进度"):
    # 添加数据结构校验
    if not isinstance(sample, dict):
        logging.error(f"样本 {sample_id} 数据结构无效: {type(sample)}")
        continue

    # 修改：demo.json中对话字段为'utts'而非'utterances'，且是字典结构
    utts = sample.get("utts", {})
    dialogue_lines = []
    for utt_id, utt in utts.items():
        speaker = utt.get("speaker", "Unknown")
        emotion = utt.get("emotion", ["Neutral"])  # 取所有情感
        text = utt.get("utterance", "")  # demo.json中文本字段为'utterance'
        dialogue_lines.append(f"{speaker} {emotion}: {text}")
    dialogue_content = "\n".join(dialogue_lines)  # 对话内容按行拼接

    # 获取 rationale 信息
    rationale = sample.get("rationale", {})

    # 新增：提取 target_change_emos 中的开始和结束情感
    target_change_emos = sample.get("target_change_emos", [])
    start_emo = None
    end_emo = None
    if len(target_change_emos) >= 2:
        # 开始emo：第一个列表的第一个元素
        if len(target_change_emos[0]) > 0:
            start_emo = target_change_emos[0][0]
        # 结束emo：第二个列表的最后一个元素
        if len(target_change_emos[1]) > 0:
            end_emo = target_change_emos[1][-1]
    if not start_emo or not end_emo:
        logging.warning(
            f"样本 {sample_id} 缺少有效的 target_change_emos: {target_change_emos}"
        )

    # 存储数据（新增：添加start_emo和end_emo到元数据）
    all_texts.append(dialogue_content)  # 用于生成嵌入向量的文本
    all_metadata.append(
        {
            "sample_id": sample_id,
            "dialogue": dialogue_content,
            "rationale": rationale,
            "start_emo": start_emo,
            "end_emo": end_emo,
        }
    )  # 元数据包含样本ID、对话、rationale和情感信息


# 定义特征提取函数
def get_text_embedding(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze(0).cpu().numpy()


# 提取所有文本的特征（新增：使用tqdm显示特征提取进度）
logging.info(f"开始提取文本特征，总文本数：{len(all_texts)}")
all_embeddings = []
for text in tqdm(all_texts, desc="特征提取进度"):  # 关键修改：用tqdm包裹循环
    embedding = get_text_embedding(text)
    all_embeddings.append(embedding)

# 转换为 numpy 数组
embeddings_np = np.array(all_embeddings).astype("float32")

# L2 归一化
faiss.normalize_L2(embeddings_np)

# 获取特征维度
D_index = embeddings_np.shape[1]

# 创建索引
index = faiss.IndexFlatIP(D_index)

# 添加向量到索引
index.add(embeddings_np)

# 保存索引
idx_path = os.path.join(workspace_root, "data/FAISS")

os.makedirs(idx_path, exist_ok=True)  # 创建父目录（若不存在）
faiss.write_index(index, os.path.join(idx_path, "faiss_stimulus_index.idx"))

# 保存对应的完整元数据（对话+rationale）到 JSON 文件（修改：存储增强后的元数据）
rationale_path = os.path.join(
    workspace_root, "data/FAISS", "faiss_stimulus_rationale_data.json"
)
rationale_dir = os.path.dirname(rationale_path)
os.makedirs(rationale_dir, exist_ok=True)
with open(rationale_path, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=4)  # 序列化增强后的元数据

logging.info(f"FAISS 索引构建完成。索引大小：{index.ntotal}，维度：{index.d}")
logging.info(
    f"索引文件: faiss_stimulus_index.idx, 元数据文件: faiss_stimulus_rationale_data.json"
)
