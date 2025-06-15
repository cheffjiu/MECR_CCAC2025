import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # 关闭tokenizer的并行化，避免与dataloader的多线程冲突，导致死锁
)


current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path)))
sys.path.extend(
    os.path.join(project_root, sub) for sub in ["", "model", "MECE_data", "utils"]
)
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from dataclasses import dataclass
from accelerate import Accelerator


# Import PEFT modules
from peft import LoraConfig, get_peft_model, TaskType

# 导入自定义模块
from model.Qwen_with_Injection import QwenWithInjection
from MECE_data.mecr_dataset import MECRDataset
from MECE_data.collate_to_graph_batch import CustomCollate
from MECE_data.build_emotion_graph import build_emotion_graph

from utils.rationale_evaluate import RationaleEvaluator

class TrainerStage1:
    def __init__(self, cfg: dataclass):
        """
        [阶段一] 训练器初始化：只为训练GNN和InjectionModule。
        """
        # 1. 初始化 Accelerator (保持不变)
        self.accelerator = Accelerator()
        self.config = cfg
        self.device = self.accelerator.device
        print(f"===== STAGE 1 - DEVICE: {self.device} =====")

        # 2. Tokenizer (保持不变)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.cfg_model.llm_tokenizer_name, 
            padding_side='left',
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. 初始化模型 (保持不变)
        qwen_base_model = AutoModelForCausalLM.from_pretrained(
            self.config.cfg_model.llm_name,
            torch_dtype=torch.bfloat16, # 使用 float32 以获得更稳定的梯度
            trust_remote_code=True,
        )
        qwen_base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = QwenWithInjection(
            qwen_base_model,
            self.config.cfg_model,
        )

        
        # 【阶段一关键修改】: 冻结所有非 GNN/Injection 的参数
        print("\n--- [STAGE 1] Configuring Model for GNN/Injection Training ---")
        # 首先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 然后只解冻 GNN 和 InjectionModule
        for name, param in self.model.named_parameters():
            if "multimodal_emotion_gnn" in name or "injection_module" in name:
                param.requires_grad = True
        
        # 打印可训练参数，确认只有 GNN 和 InjectionModule 是可训练的
        self.model.print_trainable_parameters()
        # 注意：不再需要 get_peft_model(self.model, lora_config)，因为本阶段不使用LoRA

        # 4. 数据集和数据加载器 (保持不变)
        self.train_dataset = MECRDataset(
            json_path=self.config.cfg_dataset_dataloader.json_path_train,
            feature_root=self.config.cfg_dataset_dataloader.feature_root_train,
            mode="train",
            tokenizer=self.config.cfg_dataset_dataloader.tokenizer_name,
            bert_model=self.config.cfg_dataset_dataloader.bert_name,
        )
        self.eval_dataset = MECRDataset(
            json_path=self.config.cfg_dataset_dataloader.json_path_demo,
            feature_root=self.config.cfg_dataset_dataloader.feature_root_demo,
            mode="val",
            tokenizer=self.config.cfg_dataset_dataloader.tokenizer_name,
            bert_model=self.config.cfg_dataset_dataloader.bert_name,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.cfg_dataset_dataloader.batch_size,
            shuffle=True,
            collate_fn=CustomCollate(build_emotion_graph),
            num_workers=self.config.cfg_dataset_dataloader.num_workers,
            pin_memory=True,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.cfg_dataset_dataloader.batch_size,
            shuffle=False,
            collate_fn=CustomCollate(build_emotion_graph),
            num_workers=self.config.cfg_dataset_dataloader.num_workers,
            pin_memory=True,
        )


        # 5. 优化器 
        trainable_params = [
            param for name, param in self.model.named_parameters() if param.requires_grad
        ]
        
        # 使用一个相对较高的学习率
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.config.cfg_train.stage1_learning_rate,
            weight_decay=self.config.cfg_train.stage1_weight_decay
        )

        # 【阶段一关键修改】: 创建匹配阶段一的调度器
        num_epochs = self.config.cfg_train.stage1_epochs 
        num_training_steps = (len(self.train_dataloader) // self.config.cfg_train.accumulation_steps * num_epochs)
        num_warmup_steps = int(num_training_steps * self.config.cfg_train.warmup_ratio)
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )

        # 6. 使用 accelerator.prepare() (保持不变)
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

        # 【阶段一关键修改】: 简化早停和保存逻辑
        self.evaluator = RationaleEvaluator(model_name=self.config.cfg_model.llm_name)
        self.best_eval_score = float("-inf") # 我们现在关心的是分数，不再是损失
        self.epochs_no_improve = 0
        self.patience = self.config.cfg_train.patience
        self.output_dir = self.config.cfg_train.model_save_path
        os.makedirs(self.output_dir, exist_ok=True)
        # 定义阶段一的权重保存路径
        self.stage1_gnn_path = os.path.join(self.output_dir, "stage1_gnn.pt")
        self.stage1_injection_path = os.path.join(self.output_dir, "stage1_injection.pt")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(
            self.train_dataloader,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(progress_bar):
            batched_graph = batch[0]
            prompt_texts = batch[1]
            label_texts = batch[2]
            # print(f"prompt 文本原文:{prompt_texts},形状:{len(prompt_texts)}")
            # print(f"label 文本原文:{label_texts},形状:{len(label_texts)}")
            # --- 步骤 1: 准备输入和长度信息 ---
            prompt_tokenized = self.tokenizer(prompt_texts, add_special_tokens=False)
            label_tokenized = self.tokenizer(
                [l + self.tokenizer.eos_token for l in label_texts], add_special_tokens=False
            )
            # print(f"prompt 分词后的数据：{prompt_tokenized}")
            # print(f"label 分词后的数据:{label_tokenized}")
            prompt_lens = [len(p) for p in prompt_tokenized['input_ids']]
            # print(f"prompt_lenght:{prompt_lens}")
            input_ids_list = [
                p + l for p, l in zip(prompt_tokenized['input_ids'], label_tokenized['input_ids'])
            ]
            # print(f"给Qwen3的inputs_ids:{input_ids_list}")

            # --- 步骤 2: 填充和张量化 ---
            padded_result = self.tokenizer.pad(
                {'input_ids': input_ids_list},
                padding='longest',
                max_length=1024,
                return_tensors='pt',
            ).to(self.device)
            input_ids = padded_result.input_ids
            attention_mask = padded_result.attention_mask
            # print(f"prompt+label分词后token_ids的形状：{input_ids.shape}")
            # print(f"prompt+label分词后attention_mask的形状：{attention_mask.shape}")

            # --- 步骤 3: 构建 labels ---
            labels = input_ids.clone()
            for i in range(len(labels)):
                padding_len = (attention_mask[i] == 0).sum().item()
                mask_len = padding_len + prompt_lens[i]
                if mask_len < labels.shape[1]:
                    labels[i, :mask_len] = -100

            # ====================================================================
            # =======================  第一级调试代码开始  =========================
            # ====================================================================
            # if batch_idx == 0 and self.accelerator.is_main_process:
            #     print("\n\n========================= START OF BATCH 0 DEBUG (NEW LOGIC) =========================")
            #     # 选择第一个样本进行深入分析
            #     i = 0
                
            #     # --- 1. 原始文本 ---
            #     print("\n[1.1] RAW PROMPT TEXT:")
            #     print(prompt_texts[i])
            #     print("-" * 20)
            #     print("\n[1.2] RAW LABEL TEXT:")
            #     print(label_texts[i])
            #     print("-" * 20)
                
            #     # --- 2. 拼接与编码后的结果 ---
            #     print("\n[2.1] FULL INPUT DECODED (from input_ids):")
            #     full_decoded = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
            #     print(full_decoded)
            #     print("-" * 20)

            #     # --- 3. 标签屏蔽逻辑验证 ---
            #     padding_len = (attention_mask[i] == 0).sum().item()
            #     prompt_len = prompt_lens[i]
            #     mask_len = padding_len + prompt_len
            #     print(f"\n[3.1] Calculated lengths for sample {i}: padding_len={padding_len}, prompt_len={prompt_len}, total_mask_len={mask_len}")
                
            #     # 解码被屏蔽为-100的部分 (应该是 Padding + Prompt)
            #     masked_part_ids = input_ids[i][labels[i] == -100]
            #     decoded_masked_part = self.tokenizer.decode(masked_part_ids, skip_special_tokens=False)
            #     print("\n[3.2] DECODED MASKED PART (what model should IGNORE for loss):")
            #     print(decoded_masked_part)
            #     print("-" * 20)

            #     # 解码需要计算损失的部分 (应该是 Label + EOS)
            #     unmasked_part_ids = input_ids[i][labels[i] != -100]
            #     decoded_unmasked_part = self.tokenizer.decode(unmasked_part_ids, skip_special_tokens=False)
            #     print("\n[3.3] DECODED UNMASKED PART (what model should LEARN for loss):")
            #     print(decoded_unmasked_part)
            #     print("-" * 20)
                
            #     # “标准答案”：直接对原始label文本进行分词
            #     original_label_plus_eos = label_texts[i] + self.tokenizer.eos_token
            #     decoded_original_label = self.tokenizer.decode(
            #         self.tokenizer(original_label_plus_eos, add_special_tokens=False)['input_ids'],
            #         skip_special_tokens=False
            #     )
            #     print("\n[3.4] FOR COMPARISON: ORIGINAL LABEL TOKENIZED AND DECODED:")
            #     print(decoded_original_label)
            #     print("-" * 20)

            #     # --- 4. 最终断言 ---
            #     # 移除解码后可能产生的空格或特殊前缀，进行更鲁棒的比较
            #     if decoded_unmasked_part.strip() == decoded_original_label.strip():
            #         print("\n[4.1] VERDICT: SUCCESS! The masking logic appears to be correct. [3.3] matches [3.4].")
            #     else:
            #         print("\n[4.1] VERDICT: FAILED! The masking logic is flawed. [3.3] does NOT match [3.4].")
            #     print("========================= END OF BATCH 0 DEBUG (NEW LOGIC) =========================")
            # ====================================================================
            # ========================  第一级调试代码结束  ========================
            # ====================================================================


            # ====================================================================
            # =======================  第二级调试代码开始  =========================
            # ====================================================================
            # if batch_idx == 0 and self.accelerator.is_main_process:
            #     print("\n\n========================= START OF BATCH 0 DEBUG (TENSOR-LEVEL) =========================")
            #     # 选择第一个样本进行深入分析
            #     i = 0
                
            #     # 打印基础信息
            #     prompt_len = prompt_lens[i]
            #     padding_len = (attention_mask[i] == 0).sum().item()
            #     mask_len = padding_len + prompt_len
            #     total_len = len(input_ids[i])
                
            #     print(f"\n--- SAMPLE 0 METADATA ---")
            #     print(f"Prompt Text Length (tokens): {prompt_len}")
            #     print(f"Padding Length (tokens): {padding_len}")
            #     print(f"Total Mask Length (Padding + Prompt): {mask_len}")
            #     print(f"Total Sequence Length: {total_len}")
            #     print("-" * 25)
                
            #     print("\n--- TOKEN-BY-TOKEN BREAKDOWN ---")
            #     header = f"{'Index':<8} | {'Token ID':<10} | {'Decoded Token':<30} | {'Label':<10}"
            #     print(header)
            #     print("-" * len(header))

            #     # 遍历序列中的每一个 token
            #     for j in range(total_len):
            #         token_id = input_ids[i, j].item()
            #         # 解码单个token，clean_up_tokenization_spaces=False可以保留前缀空格等信息，更真实
            #         decoded_token = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            #         label_val = labels[i, j].item()
                    
            #         # 格式化输出行
            #         row_str = f"{j:<8} | {token_id:<10} | {repr(decoded_token):<30} | {label_val:<10}"
            #         print(row_str)
                    
            #         # 在 prompt 和 label 的边界处插入一个清晰的分隔符
            #         if j == mask_len - 1:
            #             boundary_marker = f"{'':<8} | {'-'*10} | {'---- MASK ENDS / LABEL BEGINS ----':<30} | {'-'*10}"
            #             print(boundary_marker)
                        
            #     # 最后，做一个快速的完整解码对比，以防万一
            #     print("-" * 25)
            #     print("\n--- FINAL DECODED TEXTS ---")
            #     unmasked_part_ids = input_ids[i][labels[i] != -100]
            #     decoded_unmasked_part = self.tokenizer.decode(unmasked_part_ids, skip_special_tokens=False)
            #     print("\n[A] DECODED UNMASKED PART (what model should LEARN):")
            #     print(repr(decoded_unmasked_part))
                
            #     original_label_plus_eos = label_texts[i] + self.tokenizer.eos_token
            #     print("\n[B] ORIGINAL LABEL + EOS:")
            #     print(repr(original_label_plus_eos))

            #     if decoded_unmasked_part.strip() == original_label_plus_eos.strip():
            #         print("\nVERDICT: SUCCESS! Decoded parts match.")
            #     else:
            #         print("\nVERDICT: FAILED! Decoded parts DO NOT match. Check the table above for discrepancies.")

            #     print("========================= END OF BATCH 0 DEBUG (TENSOR-LEVEL) =========================")
            # ====================================================================
            # ========================  第二级调试代码结束  ========================
            # ====================================================================

            # --- 步骤 4: 模型前向传播和优化 ---
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batched_graph=batched_graph,
                labels=labels,
            )

            loss = outputs.loss

            self.accelerator.backward(loss)
            
            if self.config.cfg_train.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.cfg_train.max_grad_norm)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    def _evaluate(self):
        self.model.eval()
        all_predictions_ids = []
        all_loss = []
        all_label_texts_for_eval = []

        eval_progress_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating", # 描述可以通用一些
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch_data in enumerate(eval_progress_bar):
            with torch.no_grad():
                batched_graph, prompt_texts, label_texts_for_supervision = batch_data
                
                batched_graph = batched_graph.to(self.device)
                unwrapped_model = self.accelerator.unwrap_model(self.model)

                # --- 1. 生成部分 (回归标准调用) ---
                # 我们不再需要手动构建inputs_embeds，模型内部的forward会处理
                encoded_prompts = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=1024,
                ).to(self.device)
                
                input_ids_for_gen = encoded_prompts.input_ids
                attention_mask_for_gen = encoded_prompts.attention_mask

                
                generated_ids = unwrapped_model.generate(
                    input_ids=input_ids_for_gen,
                    attention_mask=attention_mask_for_gen,
                    batched_graph=batched_graph, # 将自定义参数传入
                    # 重要的生成控制参数
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id, 
                    max_new_tokens=self.config.cfg_train.max_new_tokens,
                    
                    # 其他beam search/sampling参数
                    num_beams=self.config.cfg_train.num_beams,
                    do_sample=self.config.cfg_train.do_sample,
                    temperature=self.config.cfg_train.temperature,
                    top_p=self.config.cfg_train.top_p,
                    top_k=self.config.cfg_train.top_k,
                    repetition_penalty=self.config.cfg_train.repetition_penalty,
                )
                
                # generate()返回的是完整的 "Prompt_IDs + Generated_IDs"
                # 所以我们从输入prompt的长度之后开始切片
                prompt_length = input_ids_for_gen.shape[1]
                sliced_generated_ids = [gen_ids[prompt_length:].tolist() for gen_ids in generated_ids]
                all_predictions_ids.extend(sliced_generated_ids)

                # --- 2. 损失计算部分 (Loss Calculation) ---
                # a. 拼接ID (与训练时相同)
                prompt_tokenized = self.tokenizer(prompt_texts, add_special_tokens=False)
                label_tokenized = self.tokenizer([l + self.tokenizer.eos_token for l in label_texts_for_supervision], add_special_tokens=False)
                input_ids_list = [p + l for p, l in zip(prompt_tokenized['input_ids'], label_tokenized['input_ids'])]

                # b. Pad (与训练时相同)
                padded_result = self.tokenizer.pad(
                    {'input_ids': input_ids_list},
                    padding='longest', max_length=1024, return_tensors='pt'
                ).to(self.device)
                full_input_ids_for_loss = padded_result.input_ids
                full_attention_mask_for_loss = padded_result.attention_mask

                # c. 创建只屏蔽了文本Prompt的labels (与训练时相同)
                labels_with_prompt_mask = full_input_ids_for_loss.clone()
                prompt_lens = [len(p) for p in prompt_tokenized['input_ids']]
                for i in range(len(labels_with_prompt_mask)):
                    padding_len = (full_attention_mask_for_loss[i] == 0).sum().item()
                    mask_len = padding_len + prompt_lens[i]
                    if mask_len < labels_with_prompt_mask.shape[1]:
                        labels_with_prompt_mask[i, :mask_len] = -100

                # d. 调用模型 (forward函数会自动处理GNN部分的屏蔽)
                outputs = self.model(
                    input_ids=full_input_ids_for_loss, 
                    attention_mask=full_attention_mask_for_loss,
                    batched_graph=batched_graph, 
                    labels=labels_with_prompt_mask # 传入这个只屏蔽了prompt的label
                )
                all_loss.append(outputs.loss.item())

                cleaned_label_texts = [txt.strip() if txt and txt.strip() else "[EMPTY]" for txt in label_texts_for_supervision]
                all_label_texts_for_eval.extend(cleaned_label_texts)

        # --- 3. 指标聚合与计算 (Metrics Aggregation) ---
        metrics = {}
        # 使用 accelerator 聚合所有GPU上的结果
        all_predictions_gathered = self.accelerator.gather_for_metrics(all_predictions_ids)
        all_references_gathered = self.accelerator.gather_for_metrics(all_label_texts_for_eval)
        all_loss_gathered = self.accelerator.gather_for_metrics(all_loss)
        
        # 只有主进程进行解码和指标计算
        if self.accelerator.is_main_process:
            # 解码所有预测的token IDs
            decoded_predictions = self.evaluator.decode_generated_tokens(all_predictions_gathered)
            
            # 打印一些样本以供调试
            print(f"\n--- DEBUG (Stage 1): Formatted Texts for Metrics ---")
            print(f"Sample decoded_predictions (first 5): {decoded_predictions[:5]}")
            print(f"Sample all_references_gathered (first 5): {all_references_gathered[:5]}")
            
            # 计算生成指标 (METEOR, BERTScore等)
            metrics = self.evaluator.compute_metrics(
                predictions=decoded_predictions, references=all_references_gathered,
            )
            # 计算并添加平均验证损失
            avg_eval_loss = torch.tensor(all_loss_gathered).mean().item()
            metrics["eval_loss"] = round(avg_eval_loss, 4)

        return metrics if self.accelerator.is_main_process else {}
    def train(self):
        # 【生成验证修改 - C】: 训练循环和保存逻辑基于 score_sum
        for epoch in range(self.config.cfg_train.stage1_epochs):
            if self.accelerator.is_main_process:
                print(f"\n--- STAGE 1: Epoch {epoch + 1}/{self.config.cfg_train.stage1_epochs} ---")

            train_loss = self._train_epoch()
            if self.accelerator.is_main_process:
                print(f"训练损失: {train_loss:.4f}")

            self.accelerator.wait_for_everyone()
            eval_metrics = self._evaluate()

            if self.accelerator.is_main_process:
                print(f"验证指标: {eval_metrics}")
                
                current_eval_score = eval_metrics.get("score_sum", float("-inf"))

                if current_eval_score > self.best_eval_score + self.config.cfg_train.min_delta:
                    print(f"验证分数改善 ({self.best_eval_score:.4f} -> {current_eval_score:.4f})，保存最佳GNN和Injection模块...")
                    self.best_eval_score = current_eval_score
                    self.epochs_no_improve = 0

                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.multimodal_emotion_gnn.state_dict(), self.stage1_gnn_path)
                    torch.save(unwrapped_model.injection_module.state_dict(), self.stage1_injection_path)
                    print(f"Stage 1 最佳权重已保存。")
                else:
                    self.epochs_no_improve += 1
                    print(f"验证损失未改善。连续无改善 Epoch 数: {self.epochs_no_improve}/{self.patience}")

                if self.epochs_no_improve >= self.patience:
                    print("早停触发！")
                    self.accelerator.end_training()
                    break
            
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print("\n--- 阶段一训练结束 ---")
            print(f"最终最佳GNN和Injection模块（score_sum: {self.best_eval_score:.4f}）已保存到:\n - {self.stage1_gnn_path}\n - {self.stage1_injection_path}")import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # 关闭tokenizer的并行化，避免与dataloader的多线程冲突，导致死锁
)


current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path)))
sys.path.extend(
    os.path.join(project_root, sub) for sub in ["", "model", "MECE_data", "utils"]
)
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from dataclasses import dataclass
from accelerate import Accelerator


# Import PEFT modules
from peft import LoraConfig, get_peft_model, TaskType

# 导入自定义模块
from model.Qwen_with_Injection import QwenWithInjection
from MECE_data.mecr_dataset import MECRDataset
from MECE_data.collate_to_graph_batch import CustomCollate
from MECE_data.build_emotion_graph import build_emotion_graph

from utils.rationale_evaluate import RationaleEvaluator

class TrainerStage1:
    def __init__(self, cfg: dataclass):
        """
        [阶段一] 训练器初始化：只为训练GNN和InjectionModule。
        """
        # 1. 初始化 Accelerator (保持不变)
        self.accelerator = Accelerator()
        self.config = cfg
        self.device = self.accelerator.device
        print(f"===== STAGE 1 - DEVICE: {self.device} =====")

        # 2. Tokenizer (保持不变)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.cfg_model.llm_tokenizer_name, 
            padding_side='left',
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. 初始化模型 (保持不变)
        qwen_base_model = AutoModelForCausalLM.from_pretrained(
            self.config.cfg_model.llm_name,
            torch_dtype=torch.float32, # 使用 float32 以获得更稳定的梯度
            trust_remote_code=True,
        )
        qwen_base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = QwenWithInjection(
            qwen_base_model,
            self.config.cfg_model,
        )

        
        # 【阶段一关键修改】: 冻结所有非 GNN/Injection 的参数
        print("\n--- [STAGE 1] Configuring Model for GNN/Injection Training ---")
        # 首先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 然后只解冻 GNN 和 InjectionModule
        for name, param in self.model.named_parameters():
            if "multimodal_emotion_gnn" in name or "injection_module" in name:
                param.requires_grad = True
        
        # 打印可训练参数，确认只有 GNN 和 InjectionModule 是可训练的
        self.model.print_trainable_parameters()
        # 注意：不再需要 get_peft_model(self.model, lora_config)，因为本阶段不使用LoRA

        # 4. 数据集和数据加载器 (保持不变)
        self.train_dataset = MECRDataset(
            json_path=self.config.cfg_dataset_dataloader.json_path_train,
            feature_root=self.config.cfg_dataset_dataloader.feature_root_train,
            mode="train",
            tokenizer=self.config.cfg_dataset_dataloader.tokenizer_name,
            bert_model=self.config.cfg_dataset_dataloader.bert_name,
        )
        self.eval_dataset = MECRDataset(
            json_path=self.config.cfg_dataset_dataloader.json_path_demo,
            feature_root=self.config.cfg_dataset_dataloader.feature_root_demo,
            mode="val",
            tokenizer=self.config.cfg_dataset_dataloader.tokenizer_name,
            bert_model=self.config.cfg_dataset_dataloader.bert_name,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.cfg_dataset_dataloader.batch_size,
            shuffle=True,
            collate_fn=CustomCollate(build_emotion_graph),
            num_workers=self.config.cfg_dataset_dataloader.num_workers,
            pin_memory=True,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.cfg_dataset_dataloader.batch_size,
            shuffle=False,
            collate_fn=CustomCollate(build_emotion_graph),
            num_workers=self.config.cfg_dataset_dataloader.num_workers,
            pin_memory=True,
        )


        # 5. 优化器 
        trainable_params = [
            param for name, param in self.model.named_parameters() if param.requires_grad
        ]
        
        # 使用一个相对较高的学习率
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.config.cfg_train.stage1_learning_rate,
            weight_decay=self.config.cfg_train.stage1_weight_decay
        )

        # 【阶段一关键修改】: 创建匹配阶段一的调度器
        num_epochs = self.config.cfg_train.stage1_epochs 
        num_training_steps = (len(self.train_dataloader) // self.config.cfg_train.accumulation_steps * num_epochs)
        num_warmup_steps = int(num_training_steps * self.config.cfg_train.warmup_ratio)
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )

        # 6. 使用 accelerator.prepare() (保持不变)
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

        # 【阶段一关键修改】: 简化早停和保存逻辑
        self.evaluator = RationaleEvaluator(model_name=self.config.cfg_model.llm_name)
        self.best_eval_score = float("-inf") # 我们现在关心的是分数，不再是损失
        self.epochs_no_improve = 0
        self.patience = self.config.cfg_train.patience
        self.output_dir = self.config.cfg_train.model_save_path
        os.makedirs(self.output_dir, exist_ok=True)
        # 定义阶段一的权重保存路径
        self.stage1_gnn_path = os.path.join(self.output_dir, "stage1_gnn.pt")
        self.stage1_injection_path = os.path.join(self.output_dir, "stage1_injection.pt")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(
            self.train_dataloader,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(progress_bar):
            batched_graph = batch[0]
            prompt_texts = batch[1]
            label_texts = batch[2]
            # print(f"prompt 文本原文:{prompt_texts},形状:{len(prompt_texts)}")
            # print(f"label 文本原文:{label_texts},形状:{len(label_texts)}")
            # --- 步骤 1: 准备输入和长度信息 ---
            prompt_tokenized = self.tokenizer(prompt_texts, add_special_tokens=False)
            label_tokenized = self.tokenizer(
                [l + self.tokenizer.eos_token for l in label_texts], add_special_tokens=False
            )
            # print(f"prompt 分词后的数据：{prompt_tokenized}")
            # print(f"label 分词后的数据:{label_tokenized}")
            prompt_lens = [len(p) for p in prompt_tokenized['input_ids']]
            # print(f"prompt_lenght:{prompt_lens}")
            input_ids_list = [
                p + l for p, l in zip(prompt_tokenized['input_ids'], label_tokenized['input_ids'])
            ]
            # print(f"给Qwen3的inputs_ids:{input_ids_list}")

            # --- 步骤 2: 填充和张量化 ---
            padded_result = self.tokenizer.pad(
                {'input_ids': input_ids_list},
                padding='longest',
                max_length=1024,
                return_tensors='pt',
            ).to(self.device)
            input_ids = padded_result.input_ids
            attention_mask = padded_result.attention_mask
            # print(f"prompt+label分词后token_ids的形状：{input_ids.shape}")
            # print(f"prompt+label分词后attention_mask的形状：{attention_mask.shape}")

            # --- 步骤 3: 构建 labels ---
            labels = input_ids.clone()
            for i in range(len(labels)):
                padding_len = (attention_mask[i] == 0).sum().item()
                mask_len = padding_len + prompt_lens[i]
                if mask_len < labels.shape[1]:
                    labels[i, :mask_len] = -100

            # ====================================================================
            # =======================  第一级调试代码开始  =========================
            # ====================================================================
            # if batch_idx == 0 and self.accelerator.is_main_process:
            #     print("\n\n========================= START OF BATCH 0 DEBUG (NEW LOGIC) =========================")
            #     # 选择第一个样本进行深入分析
            #     i = 0
                
            #     # --- 1. 原始文本 ---
            #     print("\n[1.1] RAW PROMPT TEXT:")
            #     print(prompt_texts[i])
            #     print("-" * 20)
            #     print("\n[1.2] RAW LABEL TEXT:")
            #     print(label_texts[i])
            #     print("-" * 20)
                
            #     # --- 2. 拼接与编码后的结果 ---
            #     print("\n[2.1] FULL INPUT DECODED (from input_ids):")
            #     full_decoded = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
            #     print(full_decoded)
            #     print("-" * 20)

            #     # --- 3. 标签屏蔽逻辑验证 ---
            #     padding_len = (attention_mask[i] == 0).sum().item()
            #     prompt_len = prompt_lens[i]
            #     mask_len = padding_len + prompt_len
            #     print(f"\n[3.1] Calculated lengths for sample {i}: padding_len={padding_len}, prompt_len={prompt_len}, total_mask_len={mask_len}")
                
            #     # 解码被屏蔽为-100的部分 (应该是 Padding + Prompt)
            #     masked_part_ids = input_ids[i][labels[i] == -100]
            #     decoded_masked_part = self.tokenizer.decode(masked_part_ids, skip_special_tokens=False)
            #     print("\n[3.2] DECODED MASKED PART (what model should IGNORE for loss):")
            #     print(decoded_masked_part)
            #     print("-" * 20)

            #     # 解码需要计算损失的部分 (应该是 Label + EOS)
            #     unmasked_part_ids = input_ids[i][labels[i] != -100]
            #     decoded_unmasked_part = self.tokenizer.decode(unmasked_part_ids, skip_special_tokens=False)
            #     print("\n[3.3] DECODED UNMASKED PART (what model should LEARN for loss):")
            #     print(decoded_unmasked_part)
            #     print("-" * 20)
                
            #     # “标准答案”：直接对原始label文本进行分词
            #     original_label_plus_eos = label_texts[i] + self.tokenizer.eos_token
            #     decoded_original_label = self.tokenizer.decode(
            #         self.tokenizer(original_label_plus_eos, add_special_tokens=False)['input_ids'],
            #         skip_special_tokens=False
            #     )
            #     print("\n[3.4] FOR COMPARISON: ORIGINAL LABEL TOKENIZED AND DECODED:")
            #     print(decoded_original_label)
            #     print("-" * 20)

            #     # --- 4. 最终断言 ---
            #     # 移除解码后可能产生的空格或特殊前缀，进行更鲁棒的比较
            #     if decoded_unmasked_part.strip() == decoded_original_label.strip():
            #         print("\n[4.1] VERDICT: SUCCESS! The masking logic appears to be correct. [3.3] matches [3.4].")
            #     else:
            #         print("\n[4.1] VERDICT: FAILED! The masking logic is flawed. [3.3] does NOT match [3.4].")
            #     print("========================= END OF BATCH 0 DEBUG (NEW LOGIC) =========================")
            # ====================================================================
            # ========================  第一级调试代码结束  ========================
            # ====================================================================


            # ====================================================================
            # =======================  第二级调试代码开始  =========================
            # ====================================================================
            # if batch_idx == 0 and self.accelerator.is_main_process:
            #     print("\n\n========================= START OF BATCH 0 DEBUG (TENSOR-LEVEL) =========================")
            #     # 选择第一个样本进行深入分析
            #     i = 0
                
            #     # 打印基础信息
            #     prompt_len = prompt_lens[i]
            #     padding_len = (attention_mask[i] == 0).sum().item()
            #     mask_len = padding_len + prompt_len
            #     total_len = len(input_ids[i])
                
            #     print(f"\n--- SAMPLE 0 METADATA ---")
            #     print(f"Prompt Text Length (tokens): {prompt_len}")
            #     print(f"Padding Length (tokens): {padding_len}")
            #     print(f"Total Mask Length (Padding + Prompt): {mask_len}")
            #     print(f"Total Sequence Length: {total_len}")
            #     print("-" * 25)
                
            #     print("\n--- TOKEN-BY-TOKEN BREAKDOWN ---")
            #     header = f"{'Index':<8} | {'Token ID':<10} | {'Decoded Token':<30} | {'Label':<10}"
            #     print(header)
            #     print("-" * len(header))

            #     # 遍历序列中的每一个 token
            #     for j in range(total_len):
            #         token_id = input_ids[i, j].item()
            #         # 解码单个token，clean_up_tokenization_spaces=False可以保留前缀空格等信息，更真实
            #         decoded_token = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            #         label_val = labels[i, j].item()
                    
            #         # 格式化输出行
            #         row_str = f"{j:<8} | {token_id:<10} | {repr(decoded_token):<30} | {label_val:<10}"
            #         print(row_str)
                    
            #         # 在 prompt 和 label 的边界处插入一个清晰的分隔符
            #         if j == mask_len - 1:
            #             boundary_marker = f"{'':<8} | {'-'*10} | {'---- MASK ENDS / LABEL BEGINS ----':<30} | {'-'*10}"
            #             print(boundary_marker)
                        
            #     # 最后，做一个快速的完整解码对比，以防万一
            #     print("-" * 25)
            #     print("\n--- FINAL DECODED TEXTS ---")
            #     unmasked_part_ids = input_ids[i][labels[i] != -100]
            #     decoded_unmasked_part = self.tokenizer.decode(unmasked_part_ids, skip_special_tokens=False)
            #     print("\n[A] DECODED UNMASKED PART (what model should LEARN):")
            #     print(repr(decoded_unmasked_part))
                
            #     original_label_plus_eos = label_texts[i] + self.tokenizer.eos_token
            #     print("\n[B] ORIGINAL LABEL + EOS:")
            #     print(repr(original_label_plus_eos))

            #     if decoded_unmasked_part.strip() == original_label_plus_eos.strip():
            #         print("\nVERDICT: SUCCESS! Decoded parts match.")
            #     else:
            #         print("\nVERDICT: FAILED! Decoded parts DO NOT match. Check the table above for discrepancies.")

            #     print("========================= END OF BATCH 0 DEBUG (TENSOR-LEVEL) =========================")
            # ====================================================================
            # ========================  第二级调试代码结束  ========================
            # ====================================================================

            # --- 步骤 4: 模型前向传播和优化 ---
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batched_graph=batched_graph,
                labels=labels,
            )

            loss = outputs.loss

            self.accelerator.backward(loss)
            
            if self.config.cfg_train.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.cfg_train.max_grad_norm)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    def _evaluate(self):
        self.model.eval()
        all_predictions_ids = []
        all_loss = []
        all_label_texts_for_eval = []

        eval_progress_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating", # 描述可以通用一些
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch_data in enumerate(eval_progress_bar):
            with torch.no_grad():
                batched_graph, prompt_texts, label_texts_for_supervision = batch_data
                
                batched_graph = batched_graph.to(self.device)
                unwrapped_model = self.accelerator.unwrap_model(self.model)

                # --- 1. 生成部分 (回归标准调用) ---
                # 我们不再需要手动构建inputs_embeds，模型内部的forward会处理
                encoded_prompts = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=1024,
                ).to(self.device)
                
                input_ids_for_gen = encoded_prompts.input_ids
                attention_mask_for_gen = encoded_prompts.attention_mask

                
                generated_ids = unwrapped_model.generate(
                    input_ids=input_ids_for_gen,
                    attention_mask=attention_mask_for_gen,
                    batched_graph=batched_graph, # 将自定义参数传入
                    # 重要的生成控制参数
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id, 
                    max_new_tokens=self.config.cfg_train.max_new_tokens,
                    
                    # 其他beam search/sampling参数
                    num_beams=self.config.cfg_train.num_beams,
                    do_sample=self.config.cfg_train.do_sample,
                    temperature=self.config.cfg_train.temperature,
                    top_p=self.config.cfg_train.top_p,
                    top_k=self.config.cfg_train.top_k,
                    repetition_penalty=self.config.cfg_train.repetition_penalty,
                )
                
                # generate()返回的是完整的 "Prompt_IDs + Generated_IDs"
                # 所以我们从输入prompt的长度之后开始切片
                prompt_length = input_ids_for_gen.shape[1]
                sliced_generated_ids = [gen_ids[prompt_length:].tolist() for gen_ids in generated_ids]
                all_predictions_ids.extend(sliced_generated_ids)

                # --- 2. 损失计算部分 (Loss Calculation) ---
                # a. 拼接ID (与训练时相同)
                prompt_tokenized = self.tokenizer(prompt_texts, add_special_tokens=False)
                label_tokenized = self.tokenizer([l + self.tokenizer.eos_token for l in label_texts_for_supervision], add_special_tokens=False)
                input_ids_list = [p + l for p, l in zip(prompt_tokenized['input_ids'], label_tokenized['input_ids'])]

                # b. Pad (与训练时相同)
                padded_result = self.tokenizer.pad(
                    {'input_ids': input_ids_list},
                    padding='longest', max_length=1024, return_tensors='pt'
                ).to(self.device)
                full_input_ids_for_loss = padded_result.input_ids
                full_attention_mask_for_loss = padded_result.attention_mask

                # c. 创建只屏蔽了文本Prompt的labels (与训练时相同)
                labels_with_prompt_mask = full_input_ids_for_loss.clone()
                prompt_lens = [len(p) for p in prompt_tokenized['input_ids']]
                for i in range(len(labels_with_prompt_mask)):
                    padding_len = (full_attention_mask_for_loss[i] == 0).sum().item()
                    mask_len = padding_len + prompt_lens[i]
                    if mask_len < labels_with_prompt_mask.shape[1]:
                        labels_with_prompt_mask[i, :mask_len] = -100

                # d. 调用模型 (forward函数会自动处理GNN部分的屏蔽)
                outputs = self.model(
                    input_ids=full_input_ids_for_loss, 
                    attention_mask=full_attention_mask_for_loss,
                    batched_graph=batched_graph, 
                    labels=labels_with_prompt_mask # 传入这个只屏蔽了prompt的label
                )
                all_loss.append(outputs.loss.item())

                cleaned_label_texts = [txt.strip() if txt and txt.strip() else "[EMPTY]" for txt in label_texts_for_supervision]
                all_label_texts_for_eval.extend(cleaned_label_texts)

        # --- 3. 指标聚合与计算 (Metrics Aggregation) ---
        metrics = {}
        # 使用 accelerator 聚合所有GPU上的结果
        all_predictions_gathered = self.accelerator.gather_for_metrics(all_predictions_ids)
        all_references_gathered = self.accelerator.gather_for_metrics(all_label_texts_for_eval)
        all_loss_gathered = self.accelerator.gather_for_metrics(all_loss)
        
        # 只有主进程进行解码和指标计算
        if self.accelerator.is_main_process:
            # 解码所有预测的token IDs
            decoded_predictions = self.evaluator.decode_generated_tokens(all_predictions_gathered)
            
            # 打印一些样本以供调试
            print(f"\n--- DEBUG (Stage 1): Formatted Texts for Metrics ---")
            print(f"Sample decoded_predictions (first 5): {decoded_predictions[:5]}")
            print(f"Sample all_references_gathered (first 5): {all_references_gathered[:5]}")
            
            # 计算生成指标 (METEOR, BERTScore等)
            metrics = self.evaluator.compute_metrics(
                predictions=decoded_predictions, references=all_references_gathered,
            )
            # 计算并添加平均验证损失
            avg_eval_loss = torch.tensor(all_loss_gathered).mean().item()
            metrics["eval_loss"] = round(avg_eval_loss, 4)

        return metrics if self.accelerator.is_main_process else {}
    def train(self):
        # 【生成验证修改 - C】: 训练循环和保存逻辑基于 score_sum
        for epoch in range(self.config.cfg_train.stage1_epochs):
            if self.accelerator.is_main_process:
                print(f"\n--- STAGE 1: Epoch {epoch + 1}/{self.config.cfg_train.stage1_epochs} ---")

            train_loss = self._train_epoch()
            if self.accelerator.is_main_process:
                print(f"训练损失: {train_loss:.4f}")

            self.accelerator.wait_for_everyone()
            eval_metrics = self._evaluate()

            if self.accelerator.is_main_process:
                print(f"验证指标: {eval_metrics}")
                
                current_eval_score = eval_metrics.get("score_sum", float("-inf"))

                if current_eval_score > self.best_eval_score + self.config.cfg_train.min_delta:
                    print(f"验证分数改善 ({self.best_eval_score:.4f} -> {current_eval_score:.4f})，保存最佳GNN和Injection模块...")
                    self.best_eval_score = current_eval_score
                    self.epochs_no_improve = 0

                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.multimodal_emotion_gnn.state_dict(), self.stage1_gnn_path)
                    torch.save(unwrapped_model.injection_module.state_dict(), self.stage1_injection_path)
                    print(f"Stage 1 最佳权重已保存。")
                else:
                    self.epochs_no_improve += 1
                    print(f"验证损失未改善。连续无改善 Epoch 数: {self.epochs_no_improve}/{self.patience}")

                if self.epochs_no_improve >= self.patience:
                    print("早停触发！")
                    self.accelerator.end_training()
                    break
            
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print("\n--- 阶段一训练结束 ---")
            print(f"最终最佳GNN和Injection模块（score_sum: {self.best_eval_score:.4f}）已保存到:\n - {self.stage1_gnn_path}\n - {self.stage1_injection_path}")