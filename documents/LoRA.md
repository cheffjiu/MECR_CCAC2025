非常好！你选择的两条路线都是当前多模态语言建模中最有效的方案之一。下面我将为你**分别明确这两条路径的结构设计与实施计划**，并说明它们如何协同工作。

---

## ✅ 路线 1：将 `h_change` 接入语言模型内部（Cross-Attention）

### 🎯 目标

在大语言模型（如 LLaMA、Qwen、ChatGLM 等）中插入模块，使模型能在生成 rationale 时参考图神经网络提取的情感变化向量 $h_{\text{change}} \in \mathbb{R}^D$。

---

### 📌 实现方案：插入 Cross-Attention 模块（Encoder-Decoder 风格）

#### 模块结构：

```
Embedding → Transformer Layer × N
               ↓
           Cross-Attn(h_change)
               ↓
         → 生成 rationale
```

#### 你只需：

* 冻结主干 transformer；
* 插入一个轻量模块（如 LoRA 层或 cross-attn 层）；
* 用 `h_change` 作为 KV，文本 embedding 作为 Q。

#### ✅ 插件代码（草图）：

```python
class HChangeCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, h_change):  # h_change: [B, D]
        h_change = h_change.unsqueeze(1)         # [B, 1, D]
        attn_out, _ = self.attn(hidden_states, h_change, h_change)
        return self.norm(hidden_states + attn_out)
```

#### ✅ 集成方式：

* 在 HuggingFace `transformers` 模型结构中插入此层；
* 或更简单地，在 **LoRA 微调中，接入此模块，作为独立路径训练**；
* 可只插入中间层 6/12/18 层之一（如仅对 Decoder Block 9 加）

---

## ✅ 路线 2：用 `h_change` 做检索增强 Prompt（软指导）

### 🎯 目标

用 `h_change` 向量在训练集或外部库中检索与之语义相似的样本（相似情感变化），再将其 rationale 拼入 prompt，引导语言模型生成更合理解释。

---

### 📌 实现步骤

#### ① 建立检索库

* 把训练集中所有样本的 `h_change` 保存为 `.pt` 文件；
* 预先编码为一个 FAISS index：

```python
import faiss
import torch

all_vecs = torch.load('all_h_change.pt').numpy()  # [N, D]
index = faiss.IndexFlatL2(all_vecs.shape[1])
index.add(all_vecs)
```

#### ② 查询最相近的样本（k 近邻）

```python
query = h_change.cpu().numpy().reshape(1, -1)
D, I = index.search(query, k=3)  # 得到 top-k 相似样本 index
```

#### ③ 拼接 Prompt（结构建议）：

```text
Context:
  A: ...
  B: ...
  ...

Retrieved Examples:
Example 1:
Stimulus: B表示自己不懂装修
Appraisal: A认为B工作态度不好
Response: A感到愤怒

Now explain the emotional change in this dialogue:
```

#### ✅ 构造代码模块（函数接口）

```python
def build_prompt(dialogue_context, retrieved_rationales, hint=None):
    prompt = "Context:\n" + dialogue_context + "\n\n"
    prompt += "Retrieved Examples:\n"
    for i, r in enumerate(retrieved_rationales):
        prompt += f"Example {i+1}:\n"
        prompt += f"Stimulus: {r['stimulus']['textual'] or ''}\n"
        prompt += f"Appraisal: {r['appraisal']}\n"
        prompt += f"Response: {r['response']}\n\n"
    prompt += "Now explain the emotional change in this dialogue:\n"
    return prompt
```

---

## 🔄 两者协同使用的方式：

| 阶段  | 做法                                                         |
| --- | ---------------------------------------------------------- |
| 训练时 | 输入 prompt（含上下文 + 检索样例），并同时给模型一个 cross-attn 接口接收 `h_change` |
| 推理时 | 可以仅用 `h_change` 检索出示例拼 prompt，也可一并送入模型辅助生成                 |

这样就实现了：

* ✨ 显式指令化引导（通过检索样本）
* ✨ 隐式语义控制（通过 cross-attn 融合图表示）

---

## ✅ 接下来我可以帮你：

1. ✳️ 实现 CrossAttention 插件并集成 HuggingFace LLM；
2. ✳️ 构造检索库（建立训练集 h\_change → rationale 的 FAISS 索引）；
3. ✳️ 写好 `build_prompt()` 拼接函数；
4. ✳️ 整合训练数据 → prompt → LLM + LoRA 训练器。

请告诉我你想**先做哪一步**？例如：

* ✅ “先搭好 FAISS 检索库”
* ✅ “先写 CrossAttention 层集成 HuggingFace 模型”
* ✅ “先构造 Prompt 示例用于生成调试”

你决定，我继续推进！
