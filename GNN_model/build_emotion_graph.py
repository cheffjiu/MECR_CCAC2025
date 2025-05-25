from torch_geometric.data import Data
import torch
import torch.nn as nn

def build_emotion_graph(fused_feats, utterances, change_span):
    device = fused_feats.device # 获取输入特征的设备

    # 提取说话人信息
    speakers = [utt['speaker'] for utt in utterances]

    # 创建说话人到整数的映射表
    unique_speakers = sorted(set(speakers))
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(unique_speakers)}

    # 将说话人信息转换为整数编码
    speaker_ids = [speaker_to_id[spk] for spk in speakers]

    # 创建说话人张量（使用整数编码），并确保设备一致
    spk_tensor = torch.tensor(speaker_ids, device=device).unsqueeze(1) # [N, 1]

    N = fused_feats.size(0)

    # 1. 顺序边 (前后双向)
    nodes_seq = torch.arange(N, device=device)
    src_fwd = nodes_seq[:-1]
    dst_fwd = nodes_seq[1:]
    seq_edges_fwd = torch.stack([src_fwd, dst_fwd], dim=0) # [2, N-1]
    seq_edges_bwd = torch.stack([dst_fwd, src_fwd], dim=0) # [2, N-1]
    sequential_edges = torch.cat([seq_edges_fwd, seq_edges_bwd], dim=1) # [2, 2*(N-1)]

    # 2. 同说话人边
    # 创建所有可能的边对
    row, col = torch.meshgrid(nodes_seq, nodes_seq, indexing='ij')
    all_pairs = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0) # [2, N*N]

    # 筛选出同说话人且不是自环的边
    # N = spk_tensor.shape[0]
    # spk_tensor_expanded = spk_tensor.expand(-1, N) # [N,N]
    # spk_matrix = spk_tensor_expanded == spk_tensor_expanded.T # [N, N]
    # same_spk_mask = spk_matrix
    # # old: same_spk_mask = (spk_tensor == spk_tensor.T) # [N, N]
    # same_spk_mask.fill_diagonal_(False) # 移除自环
    # src_spk, dst_spk = same_spk_mask.nonzero(as_tuple=True)
    # same_speaker_edges = torch.stack([src_spk, dst_spk], dim=0) # [2, num_same_speaker_edges]


    # 优化同说话人边创建，确保在所有节点对中筛选
    speaker_pairs_mask = (spk_tensor[all_pairs[0]] == spk_tensor[all_pairs[1]]).squeeze()
    self_loop_mask = (all_pairs[0] == all_pairs[1])

    # 过滤掉自环和非同说话人的边
    same_speaker_edges = all_pairs[:, speaker_pairs_mask & ~self_loop_mask]


    # 3. 超级节点连接
    super_idx = N # 超级节点的索引是 N，因为特征从 0 到 N-1
    # 修正 h_super: 不再是 nn.Parameter，而是普通张量，并确保在设备上
    h_super = torch.mean(fused_feats, dim=0).to(device)

    # change_span 中的索引也确保在设备上
    start_idx_tensor = torch.tensor(change_span[0], dtype=torch.long, device=device)
    end_idx_tensor = torch.tensor(change_span[1], dtype=torch.long, device=device)
    super_idx_tensor = torch.tensor(super_idx, dtype=torch.long, device=device)

    super_node_edges = torch.tensor([
        [super_idx_tensor, start_idx_tensor],
        [start_idx_tensor, super_idx_tensor],
        [super_idx_tensor, end_idx_tensor],
        [end_idx_tensor, super_idx_tensor]
    ], dtype=torch.long, device=device).t() # [2, 4]

    # 合并所有边
    edge_index = torch.cat([sequential_edges, same_speaker_edges, super_node_edges], dim=1)

    # 边属性
    # 这里的关键修改：确保属性的数量与实际边的数量一致
    num_sequential_edges = sequential_edges.size(1) # <--- **这里是修改点**
    num_same_speaker_edges = same_speaker_edges.size(1)
    num_super_node_edges = super_node_edges.size(1)

    edge_attr = torch.cat([
        torch.zeros(num_sequential_edges, 1, device=device),      # 顺序边属性0
        torch.ones(num_same_speaker_edges, 1, device=device),   # 同说话人属性1
        torch.full((num_super_node_edges, 1), 2.0, device=device) # 超级节点属性2
    ])

    # 构建Data对象
    return Data(
        x=torch.cat([fused_feats, h_super.unsqueeze(0)]), # fused_feats 和 h_super 已经在同一设备
        edge_index=edge_index, # 已经是正确的格式和设备
        edge_attr=edge_attr,
        super_idx=super_idx
    )