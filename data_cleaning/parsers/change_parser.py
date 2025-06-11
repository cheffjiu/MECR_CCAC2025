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
        # for i, (utt_pair, emos) in enumerate(zip(target_utt_ids, target_emos)):
        #     if len(utt_pair) < 2:
        #         continue
            
        #     start_utt_id, end_utt_id = utt_pair
        #     start_idx = next((idx for idx, u in enumerate(utterances) if u["id"] == start_utt_id), -1)
        #     end_idx = next((idx for idx, u in enumerate(utterances) if u["id"] == end_utt_id), -1)
            
            
        #     if start_idx == -1 or end_idx == -1:
        #         continue
            
            # # 修正情感解析逻辑
            # from_emotion = emos if isinstance(emos, list) else [emos]
            # to_emotion = emos if isinstance(emos, list) else [emos]
        start_idx = int(target_utt_ids[0][0].split("_")[-1])-1
        end_idx = int(target_utt_ids[1][-1].split("_")[-1])-1
        from_emotion = target_emos[0]
        to_emotion = target_emos[1]
        change_intervals.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "from_emotion": from_emotion,
                "to_emotion": to_emotion
            })
        
        return change_intervals
