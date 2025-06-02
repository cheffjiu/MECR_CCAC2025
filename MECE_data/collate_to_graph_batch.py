from torch_geometric.data import Batch
from typing import List, Tuple, Union


def collate_to_graph_batch(
    batch: List[dict],
    fusion_model,  # CrossModalAttention
    build_graph_fn,  # build_emotion_graph
) -> Union[
    Tuple[Batch, List[str], List[str]], Tuple[Batch, List[str]]  # train/val或者test
]:
    """
    融合多个样本并构建 PyG 批图。

    Returns:
        - batched_graph: 图神经网络批图 (PyG Batch)
        - prompt_texts: Prompt 输入列表
        - label_texts: Label 输出列表（如为 test 模式，则不包含）
    """
    # 初始化图列表
    data_list = []
    # 初始化prompt列表
    prompt_texts = []
    # 初始化label列表
    label_texts = []
    # 根据第一个样本判断是否有 label
    has_label = "label" in batch[0]

    for sample in batch:
        # 1. 融合特征
        t_feats = sample["t_feats"]
        v_feats = sample["v_feats"]
        fused_feats = fusion_model(t_feats, v_feats)

        # 2. 构图
        graph = build_graph_fn(fused_feats, sample["utterances"], sample["change_span"])
        data_list.append(graph)

        # 3. 收集 prompt 和 label
        prompt_texts.append(sample["prompt"])
        if has_label:
            label_texts.append(sample["label"])

    # 4. 聚合为批图
    batched_graph = Batch.from_data_list(data_list)

    # 5. 返回
    if has_label:
        return batched_graph, prompt_texts, label_texts
    else:
        return batched_graph, prompt_texts
