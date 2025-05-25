# build_prompt.py

def format_rationale(rationale):
    stimulus_textual = rationale['stimulus'].get('textual')
    stimulus_visual = rationale['stimulus'].get('visual')
    appraisal = rationale.get('appraisal')
    response = rationale.get('response')

    # === 根据您的最新需求修改 Stimulus 格式 ===
    rationale_lines = ["Rationale:"] # 保持 Rationale: 开头

    if stimulus_textual:
        rationale_lines.append(f"Stimulus textual: {stimulus_textual}")
    else:
        # 如果文本刺激缺失，仍然打印 Stimulus textual: 但内容为空
        rationale_lines.append("Stimulus textual:")

    if stimulus_visual:
        rationale_lines.append(f"Stimulus visual: {stimulus_visual}")
    else:
        # 如果视觉刺激缺失，仍然打印 Stimulus visual: 但内容为空
        rationale_lines.append("Stimulus visual:")

    if appraisal:
        # 您提供的数据中，response 偶尔会是 appraisal，这里需要检查一下
        # 假设 appraisal 和 response 都是字符串，且 appraisal 只有一条
        rationale_lines.append(f"Appraisal: {appraisal}")
    else:
        rationale_lines.append("Appraisal:")


    if response: # 确保 response 字段存在
         rationale_lines.append(f"Response: {response}")
    else:
        rationale_lines.append("Response:")

    # 移除空行（可选，如果确保每个字段都有占位符，则不会有空行）
    # 确保没有空行，只连接有内容的行
    # return "\n".join(line for line in rationale_lines if line.strip()) # 这一行不需要了
    return "\n".join(rationale_lines) # 直接拼接即可


def concatenate_utterances(sample):
    """将对话按时间顺序拼接为：说话人[情绪]: 文本"""
    utterances = sample['utterances']
    dialogue_lines = []
    for utt in utterances:
        speaker = utt['speaker']
        emotion = utt['emotion'][0] if utt['emotion'] else "Unknown" # 确保 emotion 不为空
        text = utt['text']
        dialogue_lines.append(f"{speaker} [{emotion}]: {text}")
    return "\n".join(dialogue_lines)

def build_prompt_from_retrieval(retrieved_examples, current_sample, include_prefix=True):
    """
    构造最终 prompt（few-shot + 当前输入）
    """
    prompt_blocks = []

    for entry in retrieved_examples:
        if 'dialogue' in entry and entry['dialogue'] and 'rationale' in entry and entry['rationale']:
            block = f"""Example:
Context:
{entry['dialogue']}

{format_rationale(entry['rationale'])}
""" # 这里直接调用 format_rationale，它会生成 Rationale: 开头
            prompt_blocks.append(block.strip())

    current_dialogue = concatenate_utterances(current_sample)

    if include_prefix:
        # === 确保这里提示模型输出完整的 Stimulus textual/visual 格式 ===
        task_block = f"""Now generate rationale for the following dialogue:
{current_dialogue}

Rationale:
Stimulus textual:
Stimulus visual:
Appraisal:
Response:"""
    else:
        task_block = f"{current_dialogue}\nRationale:\nStimulus textual:\nStimulus visual:\nAppraisal:\nResponse:"

    final_prompt = "\n\n".join(prompt_blocks + [task_block])
    return final_prompt