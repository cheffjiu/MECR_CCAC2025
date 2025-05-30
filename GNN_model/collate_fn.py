# collate_fn.py

from torch_geometric.data import Batch
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_to_graph_batch(batch_list, fusion_model, graph_builder, with_prompt=False):
    batch_t_feats = [sample['t_feats'].to(fusion_model.device) for sample in batch_list]
    batch_v_feats = [sample['v_feats'].to(fusion_model.device) for sample in batch_list]
    batch_v_mask = [sample['v_mask'].to(fusion_model.device).bool() for sample in batch_list]
    batch_lengths = [len(feats) for feats in batch_t_feats]

    padded_t_feats = pad_sequence(batch_t_feats, batch_first=True)
    padded_v_feats = pad_sequence(batch_v_feats, batch_first=True)
    padded_v_mask = pad_sequence(batch_v_mask, batch_first=True)

    fused_feats_padded = fusion_model(padded_t_feats, padded_v_feats, padded_v_mask)

    graphs = []
    all_prompt_texts = [] # 收集所有样本的 prompt 文本
    all_label_texts = []  # 收集所有样本的 label 文本

    for i, sample in enumerate(batch_list):
        current_len = batch_lengths[i]
        fused_feats = fused_feats_padded[i, :current_len, :]

        graph = graph_builder(
            fused_feats=fused_feats,
            utterances=sample['utterances'],
            change_span=sample['change_span']
        )

        # ==================== LLM 提示和标签处理（修正点） ====================
        if with_prompt:
            if 'rationale_llm_data' in sample and sample['rationale_llm_data'] is not None:
                # 直接获取原始文本字符串，不进行编码
                prompt_text = sample['rationale_llm_data']['prompt_text']
                label_text = sample['rationale_llm_data']['label_text']

                all_prompt_texts.append(prompt_text)
                all_label_texts.append(label_text)

                # 不再将编码后的张量附加到 graph 对象
                # graph.prompt_input_ids = ...
                # graph.prompt_attention_mask = ...
                # graph.label_input_ids = ...
                # graph.label_attention_mask = ...
            else:
                # 如果缺少 rationale_llm_data，则添加空字符串以保持列表长度一致
                all_prompt_texts.append("")
                all_label_texts.append("")
                print(f"Warning: with_prompt is True but rationale_llm_data missing for sample {sample['sample_id']}. Skipping LLM prompt generation for this sample.")
        # ====================================================\

        graphs.append(graph)

    batched_graph = Batch.from_data_list(graphs)

    # 返回批处理的图数据和原始的文本列表
    return batched_graph, all_prompt_texts, all_label_texts