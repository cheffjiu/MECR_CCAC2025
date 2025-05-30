# build_query.py

import torch
from transformers import AutoTokenizer


def process_utterances_for_query(utterances, tokenizer, bert_model, device):
    # 调试代码，查看 utterances 的格式
    # print(f"utterances type: {type(utterances)}")
    # print(f"utterances content: {utterances}")

    # 确保 utterances 是字符串列表
    if isinstance(utterances, dict):
        # 如果 utterances 是字典，提取其中的文本信息
        utterances = [
            utt.get("text", "") for utt in utterances if isinstance(utt, dict)
        ]

    inputs = tokenizer(
        utterances, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    query_vec = outputs.pooler_output.mean(dim=0)
    return query_vec
