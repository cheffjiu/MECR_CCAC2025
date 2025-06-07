from torch_geometric.data import Data
import torch


def build_emotion_graph(fused_feats, utterances):
    # 从fused_feats张量中得到device信息,保证所有的计算都在同一个设备上
    device = fused_feats.device
    # 设置图节点总数N
    N = fused_feats.size(0)

    # 提取说话人列表[A,B,A,B,...]
    speakers = [utt["speaker"] for utt in utterances]
    # 用集合去除重复的说话人,并排序
    unique_speakers = sorted(set(speakers))
    # 构建说话人到数字id的映射字典
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    # 将说话人列表转换为对应的数字id列表
    speaker_ids = [speaker_to_id[speaker] for speaker in speakers]
    

    # 位置编码 (归一化位置)  shape (N, 1)
    pos_encoding = torch.arange(N, device=device, dtype=torch.bfloat16).unsqueeze(1) / max(
        N - 1, 1
    )

    # 节点特征：融合特征 + 位置编码 shape (N, d_fused+1)
    x_utter = torch.cat([fused_feats, pos_encoding], dim=-1)  # 在最后一个维度上拼接

    # === 边1：顺序边 ===
    nodes = torch.arange(N, device=device)
    # 顺序边的初始列表
    seq_edges = []
    if N > 1:
        for i in range(N - 1):
            seq_edges.extend([[i, i + 1], [i + 1, i]])  # 双向顺序边
    # 将列表转换为PyTorch张量,并转置 shape [2, N-1] -> [N-1, 2]
    seq_edges = (
        torch.tensor(seq_edges, device=device).t()
        if seq_edges
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )

    # === 边2：同一说话人边 ===
    same_spk_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if speaker_ids[i] == speaker_ids[j]:
                same_spk_edges.extend([[i, j], [j, i]])
    same_spk_edges = (
        torch.tensor(same_spk_edges, device=device).t()
        if same_spk_edges
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )

    # === 边3：说话人切换边 ===
    switch_edges = []
    for i in range(N - 1):
        if speaker_ids[i] != speaker_ids[i + 1]:
            switch_edges.extend([[i, i + 1], [i + 1, i]])
    switch_edges = (
        torch.tensor(switch_edges, device=device).t()
        if switch_edges
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )

    # === 超级节点 ===
    super_idx = N
    # 超级节点位置信息 形状为[1,1]
    super_pos = torch.tensor([N], device=device, dtype=torch.bfloat16).unsqueeze(0)
    # 超级节点特征 形状为[1,d_fused]
    super_feat = torch.mean(fused_feats, dim=0, keepdim=True)
    # 超级节点特征：位置编码 + 特征均值 形状为[1,d_fused+1]
    h_super = torch.cat([super_pos, super_feat], dim=-1)
    # 节点特征：utterance特征 + 超级节点特征 形状为[N+1,d_fused+1]
    x = torch.cat([x_utter, h_super], dim=0)

    # 超级节点边
    start_idx = 0
    end_idx = N - 1
    super_edges = torch.tensor(
        [
            [super_idx, start_idx],
            [start_idx, super_idx],
            [super_idx, end_idx],
            [end_idx, super_idx],
        ],
        dtype=torch.long,
        device=device,
    ).t()

    # === 合并所有边 === 形状为[2, M]，M为边的总数
    edge_index = torch.cat(
        [seq_edges, same_spk_edges, switch_edges, super_edges], dim=1
    )

    # === 边属性 === 形状为[M, 1]，M为边的总数
    edge_attr = torch.cat(
        [
            torch.zeros(seq_edges.size(1), 1, device=device,dtype=torch.bfloat16),  # 顺序边：0
            torch.ones(same_spk_edges.size(1), 1, device=device,dtype=torch.bfloat16),  # 同speaker边：1
            torch.full(
                (switch_edges.size(1), 1), 2.0, device=device,dtype=torch.bfloat16
            ),  # 说话人切换边：2
            torch.full((super_edges.size(1), 1), 3.0, device=device,dtype=torch.bfloat16),  # 超级节点边：3
        ],
        dim=0,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        super_idx=torch.tensor(super_idx, device=device, dtype=torch.long),
    )
