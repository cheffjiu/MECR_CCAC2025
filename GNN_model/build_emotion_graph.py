from torch_geometric.data import Data
import torch
import torch.nn as nn

def build_emotion_graph(fused_feats, utterances, change_span):
    device = fused_feats.device
    N = fused_feats.size(0)

    # 提取说话人信息
    speakers = [utt['speaker'] for utt in utterances]
    unique_speakers = sorted(set(speakers))
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    speaker_ids = [speaker_to_id[spk] for spk in speakers]
    spk_tensor = torch.tensor(speaker_ids, device=device, dtype=torch.long).unsqueeze(1)

    # 1. 顺序边 (前后双向)
    nodes_seq = torch.arange(N, device=device, dtype=torch.long)
    if N > 1:
        src_fwd = nodes_seq[:-1]
        dst_fwd = nodes_seq[1:]
        seq_edges_fwd = torch.stack([src_fwd, dst_fwd], dim=0)
        seq_edges_bwd = torch.stack([dst_fwd, src_fwd], dim=0)
        sequential_edges = torch.cat([seq_edges_fwd, seq_edges_bwd], dim=1)
    else:
        sequential_edges = torch.empty((2, 0), device=device, dtype=torch.long)

    # 2. 同说话人边 - 修复MPS兼容性问题
    same_speaker_edges = []
    for i in range(N):
        for j in range(i + 1, N):  # 避免重复和自环
            if speaker_ids[i] == speaker_ids[j]:
                # 双向边
                same_speaker_edges.extend([[i, j], [j, i]])
    
    if same_speaker_edges:
        same_speaker_edges = torch.tensor(same_speaker_edges, device=device, dtype=torch.long).t()
    else:
        same_speaker_edges = torch.empty((2, 0), device=device, dtype=torch.long)

    # 3. 超级节点连接
    super_idx = N
    h_super = torch.mean(fused_feats, dim=0).to(device)

    # 确保change_span的索引在有效范围内
    start_idx = max(0, min(change_span[0], N-1))
    end_idx = max(0, min(change_span[1], N-1))
    
    super_node_edges = torch.tensor([
        [super_idx, start_idx],
        [start_idx, super_idx],
        [super_idx, end_idx],
        [end_idx, super_idx]
    ], dtype=torch.long, device=device).t()

    # 合并所有边
    edge_index = torch.cat([sequential_edges, same_speaker_edges, super_node_edges], dim=1)

    # 边属性
    num_sequential_edges = sequential_edges.size(1)
    num_same_speaker_edges = same_speaker_edges.size(1)
    num_super_node_edges = super_node_edges.size(1)

    edge_attr = torch.cat([
        torch.zeros(num_sequential_edges, 1, device=device),
        torch.ones(num_same_speaker_edges, 1, device=device),
        torch.full((num_super_node_edges, 1), 2.0, device=device)
    ])

    # 构建Data对象
    return Data(
        x=torch.cat([fused_feats, h_super.unsqueeze(0)]),
        edge_index=edge_index,
        edge_attr=edge_attr,
        super_idx=torch.tensor(super_idx, device=device, dtype=torch.long)
    )