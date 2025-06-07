import torch
from torch_geometric.data import Batch
from typing import List, Tuple, Union
from build_emotion_graph import build_emotion_graph


def collate_to_graph_batch(
    batch: List[dict],
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
        t_feats = sample["t_feats"]  # shape [N, 768]
        v_feats = sample["v_feats"]  # shape [N, 512]
        # 转换为bfloat16
        t_feats = t_feats.to(torch.bfloat16)
        v_feats = v_feats.to(torch.bfloat16)
        fused_feats = torch.cat([t_feats, v_feats], dim=-1)  # shape [N, 1280]
        # 2. 构图
        graph = build_graph_fn(fused_feats, sample["utterances"])
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

class CustomCollate:
    def __init__(self, build_graph_fn):
        """
        初始化 CustomCollate 实例。
        Args:
            build_graph_fn: 用于构建图的函数（例如 build_emotion_graph）。
        """
        self.build_graph_fn = build_emotion_graph

    def __call__(self, batch: List[dict]):
        """
        当 CustomCollate 实例被调用时，它会执行数据整理逻辑。
        Args:
            batch (List[dict]): DataLoader 传递的样本批次。
        Returns:
            Union[Tuple[Batch, List[str], List[str]], Tuple[Batch, List[str]]]:
                整理后的批图、prompt 文本和可选的 label 文本。
        """
        # 调用之前定义的 collate_to_graph_batch 函数，并传入 build_graph_fn
        return collate_to_graph_batch(batch, self.build_graph_fn)