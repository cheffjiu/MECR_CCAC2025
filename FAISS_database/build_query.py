# build_query.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 移除全局加载，改为在函数内部或通过参数传入
# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
# model = AutoModel.from_pretrained('bert-base-chinese')
# model.to(device)

def process_utterances_for_query(sample, start_idx, end_idx, tokenizer, model, device):
    """
    处理指定对话范围，拼接文本并提取 BERT 特征
    Args:
        sample (dict): 从 demo_cleaned.json 读取的单个样本数据
        start_idx (int): 起始对话索引（包含）
        end_idx (int): 结束对话索引（包含）
        tokenizer: 预加载的 HuggingFace tokenizer
        model: 预加载的 HuggingFace BERT model
        device: 模型所在的设备
    Returns:
        np.ndarray: 提取的 BERT 特征向量（用于 FAISS 查询）
    """
    utterances = sample['utterances']
    if start_idx < 0 or end_idx >= len(utterances) or start_idx > end_idx:
        raise ValueError(f"索引无效：start_idx={start_idx}, end_idx={end_idx}, 总对话数={len(utterances)}")

    dialogue_parts = []
    for i in range(start_idx, end_idx + 1):
        utt = utterances[i]
        speaker = utt['speaker']
        emotion = utt['emotion'][0]
        text = utt['text']
        dialogue_parts.append(f"{speaker} [{emotion}]: {text}")
    query_text = "\n".join(dialogue_parts)

    try:
        inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()} # 确保输入在正确设备
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 [CLS] token 的隐藏状态作为句子表示
            cls_embedding = outputs.last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
    except RuntimeError as e:
        print(f"运行时错误：{e}")
        # 如果是 MPS 设备异常，可以回退到 CPU，但这里更倾向于让调用者处理设备管理
        print("请确保 BERT 模型和输入数据在相同的设备上。")
        raise e # 重新抛出异常，让上层捕获
    return cls_embedding