import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm 

import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#获取项目根目录
current_file_path=os.path.abspath(__file__)
workspace_root=os.path.abspath(os.path.join(os.path.dirname(current_file_path),".."))

# 加载 BERT 模型和分词器
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')

#切换评估模式 + 冻结参数
model.eval()  # 切换到评估模式（关闭Dropout等）
for param in model.parameters():
    param.requires_grad = False  # 禁止参数更新

# 读取 JSON 文件
json_path=os.path.join(workspace_root, "data/processed/train_cleaned/train_cleaned.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 存储所有拼接后的对话文本和对应的完整元数据（对话+rationale）
all_texts = []  # 用于生成嵌入的文本（对话内容）
all_metadata = []  # 存储对话+rationale的完整元数据

# 遍历每个样本（新增：拼接对话内容）
logging.info(f"开始处理样本数据，总样本数：{len(data)}")
for sample in tqdm(data, desc="样本处理进度"):
    # 添加数据结构校验
    if not isinstance(sample, dict):
        logging.error(f"无效数据结构: {type(sample)}")
        continue
    utterances = sample.get('utterances', [])
    dialogue_lines = []
    for utt in utterances:
        speaker = utt['speaker']
        emotion = utt['emotion'][0]  # 取第一个情感（假设主要情感）
        text = utt['text']
        dialogue_lines.append(f"{speaker} [{emotion}]: {text}")
    dialogue_content = "\n".join(dialogue_lines)  # 对话内容按行拼接
    
    # 获取 rationale 信息
    rationale = sample['rationale']
    
    # 存储数据
    all_texts.append(dialogue_content)  # 用于生成嵌入向量的文本
    all_metadata.append({
        "dialogue": dialogue_content,
        "rationale": rationale
    })  # 元数据包含对话和 rationale

# 定义特征提取函数
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
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
embeddings_np = np.array(all_embeddings).astype('float32')

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

os.makedirs(idx_path, exist_ok=True)   # 创建父目录（若不存在）
faiss.write_index(index, os.path.join(idx_path, "faiss_stimulus_index.idx"))

# 保存对应的完整元数据（对话+rationale）到 JSON 文件（修改：存储增强后的元数据）
rationale_path = os.path.join(workspace_root, "data/FAISS", "faiss_stimulus_rationale_data.json")
rationale_dir = os.path.dirname(rationale_path)
os.makedirs(rationale_dir, exist_ok=True)
with open(rationale_path, 'w', encoding='utf-8') as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=4)  # 序列化增强后的元数据

logging.info(f"FAISS 索引构建完成。索引大小：{index.ntotal}，维度：{index.d}")
logging.info(f"索引文件: faiss_stimulus_index.idx, 元数据文件: faiss_stimulus_rationale_data.json")