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


class Trainer:
    def __init__(self, cfg: dataclass):
        """
        训练器初始化。
        """
        # 1. 初始化 Accelerator
        self.accelerator = Accelerator()
        self.config = cfg
        self.device = self.accelerator.device
        print(self.device)

        # 2. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.cfg_model.llm_tokenizer_name, 
            padding_side='left',
            trust_remote_code=True

        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. 初始化模型
        qwen_base_model = AutoModelForCausalLM.from_pretrained(
            self.config.cfg_model.llm_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        qwen_base_model.config.pad_token_id =self.tokenizer.pad_token_id
        self.model = QwenWithInjection(
            qwen_base_model,
            self.config.cfg_model,
        )

        # --- LoRA Configuration for lm_head ---
        lora_config = LoraConfig(
            r=self.config.cfg_lora.lora_r,
            lora_alpha=self.config.cfg_lora.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.config.cfg_lora.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        print("\n--- 手动启用 GNN 和 InjectionModule 参数的 requires_grad ---")
        for name, param in self.model.named_parameters():
            # 检查参数名称是否属于 GNN 或 InjectionModule
            if "multimodal_emotion_gnn" in name or "injection_module" in name:
                param.requires_grad = True
                # print(f"已启用可训练: {name}, 形状: {param.shape}")
        print("--- 启用完成 ---\n")
        self.model.print_trainable_parameters()
        # --- End LoRA Configuration ---

        # 4. 数据集和数据加载器
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

        # 5. 优化器和学习率调度器数

        no_decay = ["bias", "LayerNorm.weight"]
        # --- 在这里添加打印所有 requires_grad=True 参数的代码 ---
        print("\n--- 检查模型中的所有可训练参数 (requires_grad=True) ---")
        trainable_params_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(f"可训练参数: {name}, 形状: {param.shape}")
                trainable_params_count += 1
        print(f"总计发现 {trainable_params_count} 个可训练参数。")
        print("---------------------------------------------------\n")
        # --- 打印结束 ---
        # 只需收集 self.model (QwenWithInjection，它现在包含了所有子模块) 的参数
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.cfg_train.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()  # <--- 只需要遍历 self.model
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.config.cfg_train.learning_rate
        )

        num_training_steps = (
            len(self.train_dataloader) * self.config.cfg_train.num_train_epochs
        )
        num_warmup_steps = int(num_training_steps * self.config.cfg_train.warmup_ratio)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )

        # 6. 使用 accelerator.prepare() 包装所有组件
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

        # 7. 评估器
        self.evaluator = RationaleEvaluator(model_name=self.config.cfg_model.llm_name)

        # 8. 早停机制
        self.best_eval_score = float("-inf")
        self.epochs_no_improve = 0
        self.patience = self.config.cfg_train.patience
        self.min_delta = self.config.cfg_train.min_delta
        self.output_dir = self.config.cfg_train.model_save_path
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.output_dir, "best_model.pt")

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

            # --- 步骤 1: 准备输入和长度信息 ---
            prompt_tokenized = self.tokenizer(prompt_texts, add_special_tokens=False)
            label_tokenized = self.tokenizer(
                [l + self.tokenizer.eos_token for l in label_texts], add_special_tokens=False
            )
            prompt_lens = [len(p) for p in prompt_tokenized['input_ids']]
            input_ids_list = [
                p + l for p, l in zip(prompt_tokenized['input_ids'], label_tokenized['input_ids'])
            ]

            # --- 步骤 2: 填充和张量化 ---
            padded_result = self.tokenizer.pad(
                {'input_ids': input_ids_list},
                padding='longest',
                max_length=1024,
                return_tensors='pt',
            ).to(self.device)
            input_ids = padded_result.input_ids
            attention_mask = padded_result.attention_mask

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
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch_data in enumerate(eval_progress_bar): #
            with torch.no_grad():
                is_train_val_mode = len(batch_data) == 3

                if is_train_val_mode:
                    batched_graph, prompt_texts, label_texts_for_supervision = batch_data
                else:
                    batched_graph, prompt_texts = batch_data, None
                
                batched_graph = batched_graph.to(self.device)

                # --- 1. 生成部分 (Generation) ---
                encoded_prompts_for_gen = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=1024,
                ).to(self.device)
                
                input_ids_for_gen = encoded_prompts_for_gen["input_ids"]
                attention_mask_for_gen = encoded_prompts_for_gen["attention_mask"]

                unwrapped_model = self.accelerator.unwrap_model(self.model)
                h_change, _ = unwrapped_model.multimodal_emotion_gnn(batched_graph)

                generated_ids = unwrapped_model.generate(
                    input_ids=input_ids_for_gen,
                    attention_mask=attention_mask_for_gen,
                    h_change=h_change,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=self.config.cfg_train.max_new_tokens,
                    # ... 其他生成参数
                )
                
                prompt_length = input_ids_for_gen.shape[1]
                sliced_generated_ids = [gen_ids[prompt_length:].tolist() for gen_ids in generated_ids]
                all_predictions_ids.extend(sliced_generated_ids)

                # --- 2. 损失计算部分 (Loss Calculation) ---
                if is_train_val_mode:
                    # 应用和 _train_epoch 中完全相同的、已修正的标签屏蔽逻辑
                    prompt_tokenized = self.tokenizer(prompt_texts, add_special_tokens=False)
                    label_tokenized = self.tokenizer([l + self.tokenizer.eos_token for l in label_texts_for_supervision], add_special_tokens=False)
                    prompt_lens = [len(p) for p in prompt_tokenized['input_ids']]
                    input_ids_list = [p + l for p, l in zip(prompt_tokenized['input_ids'], label_tokenized['input_ids'])]

                    padded_result = self.tokenizer.pad(
                        {'input_ids': input_ids_list},
                        padding='longest', max_length=1024, return_tensors='pt'
                    ).to(self.device)
                    
                    full_input_ids = padded_result.input_ids
                    full_attention_mask = padded_result.attention_mask

                    labels = full_input_ids.clone()
                    for i in range(len(labels)):
                        padding_len = (full_attention_mask[i] == 0).sum().item()
                        mask_len = padding_len + prompt_lens[i]
                        if mask_len < labels.shape[1]:
                            labels[i, :mask_len] = -100

                    # ====================================================================
                    # ==============  EVALUATION BATCH 0 DEBUGGING START ==============
                    # ====================================================================
                    if batch_idx == 0 and self.accelerator.is_main_process:
                        print("\n\n========================= START OF EVAL BATCH 0 DEBUG =========================")
                        i = 0 # 只检查第一个样本
                        
                        # --- 1. 原始文本 ---
                        print("\n[EVAL-1.1] RAW PROMPT TEXT:")
                        print(prompt_texts[i])
                        print("\n[EVAL-1.2] RAW LABEL TEXT:")
                        print(label_texts_for_supervision[i])
                        print("-" * 20)
                        
                        # --- 2. 标签屏蔽逻辑验证 ---
                        padding_len = (full_attention_mask[i] == 0).sum().item()
                        prompt_len = prompt_lens[i]
                        mask_len = padding_len + prompt_len
                        print(f"\n[EVAL-2.1] Calculated lengths for sample {i}: padding_len={padding_len}, prompt_len={prompt_len}, total_mask_len={mask_len}")
                        
                        masked_ids = full_input_ids[i][labels[i] == -100]
                        decoded_masked = self.tokenizer.decode(masked_ids, skip_special_tokens=False)
                        print(f"\n[EVAL-2.2] DECODED MASKED PART (for loss):\n{decoded_masked}")
                        
                        unmasked_ids = full_input_ids[i][labels[i] != -100]
                        decoded_unmasked = self.tokenizer.decode(unmasked_ids, skip_special_tokens=False)
                        print(f"\n[EVAL-2.3] DECODED UNMASKED PART (for loss):\n{decoded_unmasked}")
                        print("-" * 20)
                        
                        # --- 3. 生成结果解码 ---
                        # 解码由 model.generate() 产生的结果
                        decoded_prediction_single = self.tokenizer.decode(sliced_generated_ids[i], skip_special_tokens=True)
                        print(f"\n[EVAL-3.1] DECODED GENERATED TEXT (for metrics):\n{decoded_prediction_single}")
                        print("========================= END OF EVAL BATCH 0 DEBUG =========================")
                    # ===================================================================
                    # ================  EVALUATION BATCH 0 DEBUGGING END ================
                    # ===================================================================

                    outputs = self.model(
                        input_ids=full_input_ids, attention_mask=full_attention_mask,
                        batched_graph=batched_graph, h_change=h_change, labels=labels
                    )
                    all_loss.append(outputs.loss.item())

                    cleaned_label_texts = [txt.strip() if txt and txt.strip() else "[EMPTY]" for txt in label_texts_for_supervision]
                    all_label_texts_for_eval.extend(cleaned_label_texts)

        # --- 3. 指标聚合与计算部分 (Metrics Aggregation) ---
        metrics = {}
        all_predictions_gathered = self.accelerator.gather_for_metrics(all_predictions_ids)
        
        if all_label_texts_for_eval:
            all_references_gathered = self.accelerator.gather_for_metrics(all_label_texts_for_eval)
            all_loss_gathered = self.accelerator.gather_for_metrics(all_loss)
            
            if self.accelerator.is_main_process:
                decoded_predictions = self.evaluator.decode_generated_tokens(all_predictions_gathered)
                
                # 你之前的详细打印，这里保留
                print(f"\n--- DEBUG (Main Process): Formatted Texts for Metrics ---")
                print(f"Sample decoded_predictions (first 5): {decoded_predictions[:5]}")
                print(f"Sample all_references_gathered (first 5): {all_references_gathered[:5]}")
                
                metrics = self.evaluator.compute_metrics(
                    predictions=decoded_predictions, references=all_references_gathered,
                )
                avg_eval_loss = torch.tensor(all_loss_gathered).mean().item()
                metrics["eval_loss"] = round(avg_eval_loss, 4)

        return metrics if self.accelerator.is_main_process else {}

    def train(self):
        
        # 开始训练循环
        for epoch in range(self.config.cfg_train.num_train_epochs):
            # 打印当前 Epoch 信息
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch + 1}/{self.config.cfg_train.num_train_epochs}")

            # --- 1. 训练一个 Epoch ---
            train_loss = self._train_epoch()
            if self.accelerator.is_main_process:
                print(f"训练损失: {train_loss:.4f}")

            # --- 2. 验证和早停判断 ---
            current_epoch_num = epoch + 1
            
            # 根据配置决定是否进行验证
            if current_epoch_num >= self.config.cfg_train.start_eval_epoch:
                
                # --- 2.1. 执行评估 ---
                # 等待所有进程完成训练步骤
                self.accelerator.wait_for_everyone()
                eval_metrics = self._evaluate()

                # --- 2.2. 主进程处理结果、保存模型和早停判断 ---
                if self.accelerator.is_main_process:
                    print(f"验证指标: {eval_metrics}")

                    # 获取当前epoch的评估分数
                    current_eval_score = eval_metrics.get("score_sum", float("-inf"))

                    # 检查性能是否改善
                    if current_eval_score > self.best_eval_score + self.config.cfg_train.min_delta:
                        print(f"验证分数改善 ({self.best_eval_score:.4f} -> {current_eval_score:.4f})，保存最佳模型...")
                        self.best_eval_score = current_eval_score
                        self.epochs_no_improve = 0

                        # 定义最佳模型的保存路径
                        best_model_save_dir = os.path.join(self.output_dir, "best_model")
                        os.makedirs(best_model_save_dir, exist_ok=True)

                        # 解包以获取原始模型
                        unwrapped_model = self.accelerator.unwrap_model(self.model)

                        # 保存所有可训练的组件
                        # a. 保存 LoRA 适配器
                        print(f"保存最佳 LoRA 适配器到 {best_model_save_dir}/lora_adapters...")
                        unwrapped_model.save_pretrained(os.path.join(best_model_save_dir, "lora_adapters"))
                        # b. 保存 InjectionModule
                        print(f"保存最佳 InjectionModule 参数到 {best_model_save_dir}/injection_module.pt...")
                        torch.save(unwrapped_model.injection_module.state_dict(), os.path.join(best_model_save_dir, "injection_module.pt"))
                        # c. 保存 MultimodalEmotionGNN
                        print(f"保存最佳 MultimodalEmotionGNN 参数到 {best_model_save_dir}/multimodal_emotion_gnn.pt...")
                        torch.save(unwrapped_model.multimodal_emotion_gnn.state_dict(), os.path.join(best_model_save_dir, "multimodal_emotion_gnn.pt"))
                        
                        print(f"最佳模型组件已保存到：{best_model_save_dir}")

                    else:
                        self.epochs_no_improve += 1
                        print(f"验证分数未改善。连续无改善 Epoch 数: {self.epochs_no_improve}/{self.patience}")

                    # 触发早停
                    if self.epochs_no_improve >= self.patience:
                        print(f"早停触发！连续 {self.patience} 个 Epoch 未见改善。")
                        # 使用 self.accelerator.end_training() 来优雅地处理多进程同步和退出
                        self.accelerator.end_training()
                        break # 跳出训练循环

            else: # 如果还没到开始验证的 epoch
                if self.accelerator.is_main_process:
                    print(f"Epoch {current_epoch_num}：初步训练中，将在 Epoch {self.config.cfg_train.start_eval_epoch} 后开始验证。")

            # 在每个 epoch 结束时等待所有进程同步，这对于早停的 break 尤其重要
            self.accelerator.wait_for_everyone()

        # --- 训练循环结束 ---
        if self.accelerator.is_main_process:
            print("\n训练结束。")
            # 检查最佳模型是否曾被保存过
            if self.best_eval_score > float("-inf"):
                print(f"最终最佳模型（score_sum: {self.best_eval_score:.4f}）的所有组件已保存到：{os.path.join(self.output_dir, 'best_model')}")
            else:
                print("训练过程中没有产生有效的最佳模型。")
