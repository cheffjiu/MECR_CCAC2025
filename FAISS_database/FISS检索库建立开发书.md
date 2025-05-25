好的，我将为您详细设计 FAISS 索引构建和查询与示例提取这两个关键模块，并以设计文档的格式呈现。

---

## FAISS 检索模块详细设计文档

### 1. 引言

本文档作为 FAISS 检索模块的补充，详细描述了其两个核心操作：FAISS 索引的构建过程（离线操作）和 FAISS 查询与示例的在线提取过程。这两个过程共同支撑了多模态情感变化推理（MECR）任务中 MLLM 的 In-Context Learning 机制。

### 2. FAISS 索引构建 (FAISS Index Construction)

此阶段是离线操作，旨在预先处理训练数据，构建一个高效可检索的特征索引，用于后续的相似示例查找。

#### 2.1. 目标

* 遍历整个训练数据集 (`train.json`)。
* 为每个包含 `rationale` 的样本（或样本中的每个情绪变化事件及其对应的 `rationale`），提取其文本刺激和视觉刺激的组合特征作为索引向量。
* 构建 FAISS 索引，并存储对应的 `rationale` 完整文本，以便检索后可以获取完整的示例内容。

#### 2.2. 输入

* **`train.json`：** 训练数据集的标注文件，包含对话、情绪变化点和 `rationale` 信息。
* **`VIDEO_FRAMES_ROOT`：** 存储视频关键帧图像的根目录路径。
* **`TextEncoder` 实例：** 预加载的中文 BERT 模型（如 `hfl/chinese-roberta-wwm-ext`），用于文本编码。
* **`VisualEncoder` 实例：** 预加载的 CLIP 图像编码器，用于视觉编码。

#### 2.3. 输出

* **`faiss_stimulus_index.idx`：** 构建好的 FAISS 索引文件。
* **`faiss_stimulus_rationale_data.json`：** 存储与索引向量对应的完整 `rationale` 字符串的 JSON 文件。

#### 2.4. 流程设计

1.  **初始化：**
    * 加载 `train.json` 数据。
    * 初始化 `TextEncoder` 和 `VisualEncoder`。
    * 定义 FAISS 索引的维度 `D_index = D_text + D_visual`。
        * `D_text` 为 `TextEncoder` 的输出维度（例如 768）。
        * `D_visual` 为 `VisualEncoder` 的输出维度（例如 512 或 768）。
    * 创建两个空列表：`all_stim_embeddings` (存储所有索引向量) 和 `all_rationale_strings` (存储对应的格式化 `rationale` 字符串)。

2.  **数据遍历与特征提取：**
    * 遍历 `train.json` 中的每一个样本（例如 `sample_id` 为键）。
    * **重要：** 每个样本的 `rationale` 字段对应一个特定的情绪变化。如果 `target_change_utt_ids` 列表包含多个情绪变化事件，并且 `rationale` 列表与之一一对应，则需要循环处理每个 `rationale`。目前根据 `anjia_sample1` 的示例，`rationale` 是一个字典，且只给出了一个，我们假设每个样本的 `rationale` 字典对应其 `target_change_utt_ids` 中的**第一个**或者**唯一**的情绪变化（这需要与数据集提供者确认，或根据实际数据情况进行调整）。这里我们以处理单个 `rationale` 为例。
    * 对于当前样本的 `rationale` 字典：
        * **a) 提取文本刺激特征：**
            * `textual_stimulus_text = rationale['stimulus']['textual']`
            * `text_stim_emb = text_encoder.encode(textual_stimulus_text)` (假设 `encode` 是 `TextEncoder` 的方法)
        * **b) 提取视觉刺激特征：**
            * `visual_stimulus_info = rationale['stimulus']['visual']`
            * **如果 `visual_stimulus_info` 为 `null`：**
                * `vis_stim_emb = np.zeros(D_visual, dtype=np.float32)`
            * **如果 `visual_stimulus_info` 为图像文件名（如 `anjia_1.jpg`）：**
                * 构建完整图像路径：`image_path = os.path.join(VIDEO_FRAMES_ROOT, sample_id, visual_stimulus_info)`
                * `vis_stim_emb = visual_encoder.encode(image_path)` (假设 `encode` 是 `VisualEncoder` 的方法)
            * **兼容性处理：** 确保 `text_stim_emb` 和 `vis_stim_emb` 都为 `numpy.ndarray` 且 `dtype=np.float32`。
        * **c) 拼接索引向量：**
            * `current_index_vector = np.concatenate([text_stim_emb, vis_stim_emb])`
            * 将其添加到 `all_stim_embeddings` 列表。
        * **d) 格式化并存储 `rationale` 字符串：**
            * 获取 `appraisal_text = rationale['appraisal']`
            * 获取 `response_text = rationale['response']`
            * 根据评测要求格式化视觉刺激文本：
                * `visual_stim_for_prompt = "null"` if `visual_stimulus_info` is `null` else `visual_stimulus_info` (如果 `visual_stimulus_info` 本身就是文本描述，则直接使用；如果是图片文件名，可能需要一个预定义的占位符，如 `"[VISUAL_INFO]"`，或者忽略此部分，取决于对评测格式的理解和期望效果)。**根据 `readme.pdf`，这里如果是 `null` 就填 `null`，否则应该就是实际的视觉描述，例如图像中检测到的内容。这里我们假设 `visual_stimulus_info` 要么是 `null` 要么是文本。**
            * `full_rationale_string = f"{textual_stimulus_text}; {visual_stim_for_prompt} {appraisal_text} {response_text}"`
            * 将其添加到 `all_rationale_strings` 列表。

3.  **FAISS 索引构建：**
    * 将 `all_stim_embeddings` 转换为 `numpy.ndarray` (`embeddings_np = np.array(all_stim_embeddings).astype('float32')`)。
    * **L2 归一化：** `faiss.normalize_L2(embeddings_np)`。
    * **创建索引：** `index = faiss.IndexFlatIP(D_index)`。
    * **添加向量：** `index.add(embeddings_np)`。

4.  **保存：**
    * 将 `index` 保存到文件：`faiss.write_index(index, 'faiss_stimulus_index.idx')`。
    * 将 `all_rationale_strings` 列表保存为 JSON 文件：
        ```python
        import json
        with open('faiss_stimulus_rationale_data.json', 'w', encoding='utf-8') as f:
            json.dump(all_rationale_strings, f, ensure_ascii=False, indent=4)
        ```

#### 2.5. 伪代码示例

```python
import faiss
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image

# --- 配置 ---
TRAIN_JSON_PATH = 'train.json'
VIDEO_FRAMES_ROOT = 'path/to/your/video_frames/' # 确保视频帧已提取并组织好

# --- 特征编码器初始化 ---
# 文本编码器
text_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
text_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
text_model.eval()
D_text = text_model.config.hidden_size # 768

# 视觉编码器
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
D_visual = clip_model.config.vision_config.hidden_size # 512 or 768, depends on model

# --- 辅助函数 ---
def get_text_embedding(text):
    if not text:
        return np.zeros(D_text, dtype=np.float32)
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.pooler_output.squeeze(0).cpu().numpy()

def get_visual_embedding(image_path):
    if not image_path or not os.path.exists(image_path):
        return np.zeros(D_visual, dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs).squeeze(0)
    return embedding.cpu().numpy()

# --- FAISS 索引构建 ---
print("开始构建 FAISS 索引...")

with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

all_stim_embeddings = []
all_rationale_strings = []

D_index = D_text + D_visual # 拼接后的维度

for sample_id, sample_data in train_data.items():
    # 假设每个样本的rationale字段直接对应一个情绪变化事件
    # 如果一个样本有多个target_change_utt_ids和多个rationale，需要在此处循环处理
    rationale = sample_data['rationale'] # Simplified: assuming one rationale per sample

    # 1. 文本刺激特征
    textual_stim_text = rationale['stimulus']['textual']
    text_stim_emb = get_text_embedding(textual_stim_text)

    # 2. 视觉刺激特征
    visual_stim_info = rationale['stimulus']['visual']
    vis_stim_emb = None
    visual_stim_for_prompt = "null" # Default for prompt formatting

    if visual_stim_info is None:
        vis_stim_emb = np.zeros(D_visual, dtype=np.float32)
    else:
        # Assuming visual_stim_info is a filename like 'frame_0001.jpg' relative to sample_id dir
        image_path = os.path.join(VIDEO_FRAMES_ROOT, sample_id, visual_stim_info)
        vis_stim_emb = get_visual_embedding(image_path)
        visual_stim_for_prompt = visual_stim_info # Use filename or actual description if available

    # 3. 拼接索引向量
    current_index_vector = np.concatenate([text_stim_emb, vis_stim_emb])
    all_stim_embeddings.append(current_index_vector)

    # 4. 格式化并存储 rationale 字符串
    appraisal_text = rationale['appraisal']
    response_text = rationale['response']
    full_rationale_string = f"{textual_stim_text}; {visual_stim_for_prompt} {appraisal_text} {response_text}"
    all_rationale_strings.append(full_rationale_string)

# 转换为 numpy 数组
embeddings_np = np.array(all_stim_embeddings).astype('float32')

# L2 归一化
faiss.normalize_L2(embeddings_np)

# 创建并添加索引
index = faiss.IndexFlatIP(D_index)
index.add(embeddings_np)

# 保存索引和元数据
faiss.write_index(index, 'faiss_stimulus_index.idx')
with open('faiss_stimulus_rationale_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_rationale_strings, f, ensure_ascii=False, indent=4)

print(f"FAISS 索引构建完成。索引大小：{index.ntotal}，维度：{index.d}")
print(f"索引文件: faiss_stimulus_index.idx, 元数据文件: faiss_stimulus_rationale_data.json")
```

---

### 3. FAISS 查询与示例提取 (FAISS Query & Example Extraction)

此阶段是在 MLLM 训练的每个批次或推理时执行的在线操作，用于根据当前输入样本动态检索最相似的 In-Context Learning 示例。

#### 3.1. 目标

* 根据当前推理样本的对话上下文，生成查询向量。
* 在预构建的 FAISS 索引中高效地检索出 Top-K 个最相似的 `rationale` 示例。
* 将检索到的示例格式化为 MLLM Prompt 所需的字符串，并返回。

#### 3.2. 输入

* **当前样本数据：** 包含 `utts` (所有对话 utterance) 和 `target_change_utt_ids` (情绪变化区间) 的字典。
* **`TextEncoder` 实例：** 与索引构建时使用的相同实例。
* **`faiss_stimulus_index.idx`：** 预加载的 FAISS 索引。
* **`faiss_stimulus_rationale_data.json`：** 预加载的 `rationale` 元数据。
* **`k`：** 要检索的示例数量（例如 3-5）。

#### 3.3. 输出

* **`formatted_examples_string`：** 包含 `k` 个格式化示例的字符串，可直接拼接到 MLLM Prompt 中。

#### 3.4. 流程设计

1.  **加载资源（一次性操作）：**
    * 在模型训练/推理开始时，加载 `faiss_stimulus_index.idx` 到内存。
    * 加载 `faiss_stimulus_rationale_data.json` 到内存。
    * 确保 `TextEncoder` 实例已初始化并可用。

2.  **构建查询向量 (`query_vector`)：**
    * **a) 提取查询文本：**
        * 根据当前样本的 `target_change_utt_ids`，获取其对应的情绪变化起点和终点 utterance ID。
        * 从 `utts` 字典中，提取这些 ID 对应的完整 `utterance` 文本。
        * **策略：** 为了提供足够的上下文，建议拼接 `target_change_utt_ids` 中涉及的所有 utterance，以及其前一个或前两个 utterance（如果存在），并包含说话人信息。
        * `query_context_text = f"{speaker_A}: {utt_A_text} {speaker_B}: {utt_B_text} ..."`
    * **b) 编码查询向量：**
        * `query_vector = text_encoder.encode(query_context_text)` (确保 `encode` 方法与索引构建时一致)
        * 将其转换为 `numpy.ndarray` (`query_vector.astype('float32')`)。
        * **重塑：** `query_vector = query_vector.reshape(1, -1)` (适应 FAISS `search` 方法的批量输入)。

3.  **执行 FAISS 检索：**
    * **L2 归一化：** `faiss.normalize_L2(query_vector)` （与索引构建时保持一致）。
    * **执行搜索：** `distances, indices = faiss_index.search(query_vector, k)`。
        * `indices` 将是一个 `(1, k)` 的 NumPy 数组，包含检索到的最相似向量的索引。

4.  **提取并格式化示例：**
    * 初始化一个空字符串 `formatted_examples_string = ""`。
    * 遍历 `indices[0]` (因为 `query_vector` 是单样本查询，结果在 `indices[0]` 中)。
    * 对于每个检索到的索引 `idx`：
        * 从 `all_rationale_strings` 列表中获取 `retrieved_rationale_string = all_rationale_strings[idx]`。
        * 将此字符串添加到 `formatted_examples_string`，并遵循 Prompt 期望的格式：
            `formatted_examples_string += f"{i+1}) 完整理由: {retrieved_rationale_string}\n"`
            （其中 `i` 是当前示例的序号，从 0 开始）。

5.  **返回：** `return formatted_examples_string`。

#### 3.5. 伪代码示例

```python
import faiss
import numpy as np
import json
import os
import torch
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image

# --- 配置（应与索引构建时一致）---
VIDEO_FRAMES_ROOT = 'path/to/your/video_frames/'
FAISS_INDEX_PATH = 'faiss_stimulus_index.idx'
RATIONALE_DATA_PATH = 'faiss_stimulus_rationale_data.json'

# --- 特征编码器初始化（与构建时共享实例或重新加载） ---
text_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
text_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
text_model.eval()
D_text = text_model.config.hidden_size # 768

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
D_visual = clip_model.config.vision_config.hidden_size # 512 or 768

# --- 辅助函数 (与索引构建时一致) ---
def get_text_embedding(text):
    if not text:
        return np.zeros(D_text, dtype=np.float32)
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.pooler_output.squeeze(0).cpu().numpy()

# Visual encoder not directly used for query embedding, but defined for completeness if needed elsewhere
def get_visual_embedding(image_path):
    if not image_path or not os.path.exists(image_path):
        return np.zeros(D_visual, dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs).squeeze(0)
    return embedding.cpu().numpy()

# --- FAISS 查询模块 ---
class FAISSRetriever:
    def __init__(self, index_path, rationale_data_path):
        print(f"加载 FAISS 索引: {index_path}")
        self.faiss_index = faiss.read_index(index_path)
        print(f"加载 Rationale 元数据: {rationale_data_path}")
        with open(rationale_data_path, 'r', encoding='utf-8') as f:
            self.all_rationale_strings = json.load(f)
        self.text_encoder = get_text_embedding # Bind the function

    def retrieve_examples(self, current_sample_data, k=3):
        """
        根据当前样本数据，从FAISS索引中检索最相似的k个Rationale示例。

        Args:
            current_sample_data (dict): 包含 'utts' 和 'target_change_utt_ids' 的样本数据。
            k (int): 检索示例的数量。

        Returns:
            str: 格式化后的检索示例字符串，用于MLLM Prompt。
        """
        # 1. 构建查询文本
        utts_data = current_sample_data['utts']
        target_change_utt_ids_list = current_sample_data['target_change_utt_ids']

        # 假设我们总是检索与第一个情绪变化（如果存在多个）相关的示例
        # 实际应用中，如果每个样本只处理一个rationale，那么这里直接取第一个
        # 如果一个样本有多个rationale，需要传入是第几个变化
        target_change_utt_id_pair = target_change_utt_ids_list[0] # 例如 ['anjia_sample1_5', 'anjia_sample1_5']

        # 获取情绪变化点及前一两个utterance的文本作为查询上下文
        # 找到target_change_utt_id_pair中最后一个utterance的ID
        end_utt_id_in_pair = target_change_utt_id_pair[1]
        
        # 获取所有utterance的有序列表，便于根据索引取前后文
        ordered_utt_ids = list(utts_data.keys())
        end_utt_idx = ordered_utt_ids.index(end_utt_id_in_pair)

        # 考虑情绪变化点及其前2个utterance作为上下文 (可调)
        start_idx_for_query = max(0, end_utt_idx - 2) 
        context_utt_ids = ordered_utt_ids[start_idx_for_query : end_utt_idx + 1]

        query_context_texts = []
        for utt_id in context_utt_ids:
            utt_info = utts_data[utt_id]
            query_context_texts.append(f"{utt_info['speaker']}: {utt_info['utterance']}")
        
        query_context_text = " ".join(query_context_texts) # 拼接为查询字符串

        # 2. 编码查询向量
        query_vector = self.text_encoder(query_context_text)
        query_vector = query_vector.reshape(1, -1).astype('float32') # Reshape for FAISS
        faiss.normalize_L2(query_vector) # Normalize

        # 3. 执行检索
        distances, indices = self.faiss_index.search(query_vector, k)

        # 4. 提取并格式化示例
        formatted_examples_string = "检索到的相关示例:\n"
        for i, idx in enumerate(indices[0]):
            retrieved_rationale_string = self.all_rationale_strings[idx]
            formatted_examples_string += f"{i+1}) 完整理由: {retrieved_rationale_string}\n"
        
        return formatted_examples_string

# --- 使用示例 ---
if __name__ == "__main__":
    # 假装这是你的某个 val/test 样本数据
    # 这里使用 anjia_sample1 的数据结构作为示例
    sample_data_for_query = {
        "utts": {
            "anjia_1_1": {"utterance": "你先把衣服换了，然后去这个地址。", "speaker": "A"},
            "anjia_1_2": {"utterance": "一会儿会有装修公司的人过去，你去监工，看看有什么可以帮忙的。", "speaker": "A"},
            "anjia_1_3": {"utterance": "这个活也派我干呀？", "speaker": "B"},
            "anjia_1_4": {"utterance": "装修我不懂。", "speaker": "B"},
            "anjia_1_5": {"utterance": "你发了两天传单了，有没有意向客户啊？要到人家电话没有？", "speaker": "A"}
        },
        "target_change_utt_ids": [ # 假设我们要查询的是第二个情绪变化
            ["anjia_1_1", "anjia_1_2"], # 比如第一个变化
            ["anjia_1_5", "anjia_1_5"]  # 比如第二个变化
        ]
    }

    # 1. 运行 FAISS 索引构建代码 (确保 VIDEO_FRAMES_ROOT 设置正确，并有图像文件)
    #    此步骤会生成 faiss_stimulus_index.idx 和 faiss_stimulus_rationale_data.json
    #    可以单独运行一次脚本来完成索引构建

    # 2. 初始化检索器
    retriever = FAISSRetriever(FAISS_INDEX_PATH, RATIONALE_DATA_PATH)

    # 3. 进行检索
    retrieved_prompt_part = retriever.retrieve_examples(sample_data_for_query, k=3)
    print("\n--- 检索到的 Prompt 片段 ---")
    print(retrieved_prompt_part)
```