import json
import os
import numpy as np
from build_query import process_utterances_for_query
from faiss_query import query_similar_rationale
from build_prompt import build_prompt_from_retrieval

#项目根目录
current_file_path = os.path.abspath(__file__)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))
def load_demo_data():
    """加载demo_cleaned.json测试数据"""
    demo_path = os.path.join(workspace_root,  "data/processed/demo_cleaned/demo_cleaned.json")
    with open(demo_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_integration_workflow():
    # 加载测试数据
    demo_data = load_demo_data()
    current_sample = demo_data[2]  # 选择 anjia_sample44
    print(f"当前测试样本ID: {current_sample['sample_id']}")

    # ----------------------
    # 测试步骤1: 生成查询向量
    # ----------------------
    try:
        query_feature = process_utterances_for_query(current_sample, 0, 2)
        # 验证特征向量形状和归一化
        assert query_feature.shape == (1, 768), "BERT特征维度错误"
        norm = np.linalg.norm(query_feature)
        assert np.isclose(norm, 1.0, atol=1e-6), "特征未正确归一化"
        print("√ 步骤1: 查询向量生成成功（形状正确且已L2归一化）")
    except Exception as e:
        print(f"× 步骤1失败: {str(e)}")
        return

    # ----------------------
    # 测试步骤2: FAISS检索
    # ----------------------
    try:
        top_k = 1
        top_rationales = query_similar_rationale(query_feature, k=top_k)
        assert len(top_rationales) == top_k, "返回结果数量不符"
        assert all(isinstance(r, dict) for r in top_rationales), "返回数据格式错误"
        print(f"√ 步骤2: 成功检索到{top_k}条理性数据")

        # 输出检索结果的对话片段（代替 sample_id）
        print("检索结果示例:")
        for idx, rationale in enumerate(top_rationales):
            dialogue_lines = rationale["dialogue"].split("\n")[:3]  # 取前3行
            print(f"结果 {idx+1}:")
            print("\n".join(dialogue_lines))
            print("---")
    except Exception as e:
        print(f"× 步骤2失败: {str(e)}")
        return

    # ----------------------
    # 测试步骤3: 构建Prompt
    # ----------------------
    try:
        final_prompt = build_prompt_from_retrieval(top_rationales, current_sample)
        # 验证关键组件存在
        assert "Example:" in final_prompt, "示例块缺失"
        assert "Rationale:" in final_prompt, "理性描述块缺失"
        assert current_sample['utterances'][0]['text'] in final_prompt, "当前对话未拼接"
        print("√ 步骤3: Prompt构建成功")
        print("\n--- 最终Prompt预览（截断）---")
        print(final_prompt)  # 避免控制台溢出
    except Exception as e:
        print(f"× 步骤3失败: {str(e)}")
        return

    print("\n所有测试通过 ✅")

if __name__ == "__main__":
    # 执行测试流程
    test_integration_workflow()