from typing import List, Dict, Any

class ChangeIntervalParser:
    def parse(
        self, 
        utterances: List[Dict], 
        target_utt_ids: List[List[str]], 
        target_emos: List[List[str]]
    ) -> List[Dict[str, Any]]:
        change_intervals = []
        
        # 遍历每个情感变化对
        for i, (utt_pair, emos) in enumerate(zip(target_utt_ids, target_emos)):
            if len(utt_pair) < 2:
                continue  # 跳过无效的变化对
            
            # 获取起始和结束utterance的ID
            start_utt_id, end_utt_id = utt_pair
            # 查找在utterances列表中的索引
            start_idx = next((idx for idx, u in enumerate(utterances) if u["id"] == start_utt_id), -1)
            end_idx = next((idx for idx, u in enumerate(utterances) if u["id"] == end_utt_id), -1)
            
            if start_idx == -1 or end_idx == -1:
                continue  # 跳过索引不存在的变化对
            
            # 构造情感变化区间（假设target_emos格式为[[from_emotion], [to_emotion]]）
            from_emotion = emos[0] if len(emos) > 0 else []
            to_emotion = emos[1] if len(emos) > 1 else []  # 根据实际数据调整索引
            
            change_intervals.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "from_emotion": from_emotion,
                "to_emotion": to_emotion
            })
        
        return change_intervals
