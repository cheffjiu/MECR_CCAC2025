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


class TrainerStage2: 
    def __init__(self, cfg: dataclass):
        """
        [阶段二 - 最终版] 训练器：加载预训练的GNN，冻结所有，只微调LoRA。
        """
        # 1. 初始化 Accelerator
        self.accelerator = Accelerator()
        self.config = cfg
        self.device = self.accelerator.device
        print(f"===== STAGE 2 (LoRA-Only) - DEVICE: {self.device} =====")

        # 2. Tokenizer (不变)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.cfg_model.llm_tokenizer_name, 
            padding_side='left',
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. 初始化模型 (不变)
        qwen_base_model = AutoModelForCausalLM.from_pretrained(
            self.config.cfg_model.llm_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        qwen_base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = QwenWithInjection(
            qwen_base_model,
            self.config.cfg_model,
        )
        
        # 【最终关键修改 - A】: 加载阶段一权重并冻结GNN/Injection模块
        unwrapped_model_for_loading = self.model
        stage1_gnn_path = os.path.join(self.config.cfg_train.model_save_path, "stage1_gnn.pt")
        stage1_injection_path = os.path.join(self.config.cfg_train.model_save_path, "stage1_injection.pt")
        
        print("\n--- [STAGE 2] Loading and Freezing weights from Stage 1 ---")
        try:
            unwrapped_model_for_loading.multimodal_emotion_gnn.load_state_dict(torch.load(stage1_gnn_path, map_location='cpu'))
            unwrapped_model_for_loading.injection_module.load_state_dict(torch.load(stage1_injection_path, map_location='cpu'))
            print("Successfully loaded weights for GNN and InjectionModule.")
        except FileNotFoundError as e:
            print(f"致命错误：找不到阶段一的权重文件，无法进行阶段二训练。错误信息: {e}")
            raise e

        # 将GNN和注入模块设为评估模式并冻结其梯度
        unwrapped_model_for_loading.multimodal_emotion_gnn.eval()
        unwrapped_model_for_loading.injection_module.eval()
        for param in unwrapped_model_for_loading.multimodal_emotion_gnn.parameters():
            param.requires_grad = False
        for param in unwrapped_model_for_loading.injection_module.parameters():
            param.requires_grad = False
        print("GNN and Injection modules are now frozen.")
        
        # --- LoRA 配置和应用 (不变) ---
        lora_config = LoraConfig(
            r=self.config.cfg_lora.lora_r,
            lora_alpha=self.config.cfg_lora.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.config.cfg_lora.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # 【最终关键修改 - B】: 打印可训练参数，以验证只有LoRA是可训练的
        print("\n--- [STAGE 2] Verifying Trainable Parameters (should be LoRA only) ---")
        self.model.print_trainable_parameters()

        # 4. 数据集和数据加载器 (不变)
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

        # 5. 【最终关键修改 - C】: 创建只包含LoRA参数的优化器
        print("\n--- [STAGE 2] Creating Optimizer for LoRA-only parameters ---")
        
        # 使用filter可以非常简洁地获取所有 requires_grad=True 的参数
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        # 从配置中读取专门为LoRA设置的学习率
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.config.cfg_train.lora_learning_rate,
            weight_decay=self.config.cfg_train.weight_decay
        )

        # 【最终关键修改 - D】: 创建匹配阶段二的调度器
        num_epochs = self.config.cfg_train.stage2_epochs
        num_training_steps = (len(self.train_dataloader) // self.config.cfg_train.accumulation_steps * num_epochs)
        num_warmup_steps = int(num_training_steps * self.config.cfg_train.warmup_ratio)
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # 6. Accelerator.prepare (不变)
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
        # 7. 评估器和早停机制 (保持不变)
        self.evaluator = RationaleEvaluator(model_name=self.config.cfg_model.llm_name)
        self.best_eval_score = float("-inf")
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
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch_data in enumerate(eval_progress_bar):
            with torch.no_grad():
                is_train_val_mode = len(batch_data) == 3

                if is_train_val_mode:
                    batched_graph, prompt_texts, label_texts_for_supervision = batch_data
                else:
                    # 这种模式下无法进行损失计算，仅用于推理
                    batched_graph, prompt_texts = batch_data, None 
                
                batched_graph = batched_graph.to(self.device)
                unwrapped_model = self.accelerator.unwrap_model(self.model)

                # ================================================================
                # 【终极修复 - A】: 在 .generate() 之前，手动预计算所有 embeddings
                # ================================================================
                # 1. 获取文本的词嵌入
                encoded_prompts = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=1024, # 确保与训练时一致
                ).to(self.device)
                text_input_ids = encoded_prompts.input_ids
                text_attention_mask = encoded_prompts.attention_mask
                
                # 使用 unwrapped_model 确保我们调用的是原始模型的方法
                text_embeds = unwrapped_model.get_input_embeddings()(text_input_ids)

                # 2. 计算 GNN 特征
                h_change, _ = unwrapped_model.multimodal_emotion_gnn(batched_graph)
                gnn_embeds = unwrapped_model.injection_module(h_change)

                # 3. 拼接成最终的 inputs_embeds
                inputs_embeds = torch.cat([gnn_embeds, text_embeds], dim=1)
                
                # 4. 创建与之匹配的 attention_mask
                gnn_attention_mask = torch.ones(
                    gnn_embeds.shape[:2], dtype=torch.long, device=self.device
                )
                full_attention_mask_for_gen = torch.cat([gnn_attention_mask, text_attention_mask], dim=1)


                # ================================================================
                # 【终极修复 - B】: 调用 .generate() 时，只传入 inputs_embeds 和 attention_mask
                # ================================================================
                generated_ids = unwrapped_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention_mask_for_gen,
                    batched_graph=batched_graph,
                    pad_token_id=self.tokenizer.pad_token_id, 
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.config.cfg_train.max_new_tokens,
                    num_beams=self.config.cfg_train.num_beams,
                    do_sample=self.config.cfg_train.do_sample,
                    temperature=self.config.cfg_train.temperature,
                    top_p=self.config.cfg_train.top_p,
                    top_k=self.config.cfg_train.top_k,
                    repetition_penalty=self.config.cfg_train.repetition_penalty,
                )
                
                # 【终极修复 - C】: 切片逻辑基于 inputs_embeds 的长度
                prompt_length = inputs_embeds.shape[1]
                sliced_generated_ids = [gen_ids[prompt_length:].tolist() for gen_ids in generated_ids]
                all_predictions_ids.extend(sliced_generated_ids)

                # --- 2. 损失计算部分 (Loss Calculation) ---
                # 这部分逻辑保持不变，因为它不调用 .generate()，使用的是原始、独立的输入处理流程
                if is_train_val_mode:
                    prompt_tokenized = self.tokenizer(prompt_texts, add_special_tokens=False)
                    label_tokenized = self.tokenizer([l + self.tokenizer.eos_token for l in label_texts_for_supervision], add_special_tokens=False)
                    prompt_lens = [len(p) for p in prompt_tokenized['input_ids']]
                    input_ids_list = [p + l for p, l in zip(prompt_tokenized['input_ids'], label_tokenized['input_ids'])]

                    padded_result = self.tokenizer.pad(
                        {'input_ids': input_ids_list}, padding='longest', max_length=1024, return_tensors='pt'
                    ).to(self.device)
                    
                    full_input_ids_for_loss = padded_result.input_ids
                    full_attention_mask_for_loss = padded_result.attention_mask

                    labels = full_input_ids_for_loss.clone()
                    for i in range(len(labels)):
                        padding_len = (full_attention_mask_for_loss[i] == 0).sum().item()
                        mask_len = padding_len + prompt_lens[i]
                        if mask_len < labels.shape[1]:
                            labels[i, :mask_len] = -100
                    
                    # 注意：这里我们使用 input_ids 来计算损失，而不是 inputs_embeds
                    # 因为模型的 forward 函数在训练和损失计算时，仍然是基于 input_ids 来构建所有输入的
                    outputs = self.model(
                        input_ids=full_input_ids_for_loss, 
                        attention_mask=full_attention_mask_for_loss,
                        batched_graph=batched_graph, 
                        labels=labels
                    )
                    all_loss.append(outputs.loss.item())

                    cleaned_label_texts = [txt.strip() if txt and txt.strip() else "[EMPTY]" for txt in label_texts_for_supervision]
                    all_label_texts_for_eval.extend(cleaned_label_texts)

        # --- 3. 指标聚合与计算部分 (Metrics Aggregation) ---
        # 这部分代码完全不变
        metrics = {}
        all_predictions_gathered = self.accelerator.gather_for_metrics(all_predictions_ids)
        
        if all_label_texts_for_eval:
            all_references_gathered = self.accelerator.gather_for_metrics(all_label_texts_for_eval)
            all_loss_gathered = self.accelerator.gather_for_metrics(all_loss)
            
            if self.accelerator.is_main_process:
                decoded_predictions = self.evaluator.decode_generated_tokens(all_predictions_gathered)
                
                # DEBUG 打印可以保留，但现在我们期望它能输出有意义的内容
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
        for epoch in range(self.config.cfg_train.stage2_epochs):
            if self.accelerator.is_main_process:
                print(f"\n--- STAGE 2: Epoch {epoch + 1}/{self.config.cfg_train.stage2_epochs} ---")

            train_loss = self._train_epoch()

            if self.accelerator.is_main_process:
                print(f"训练损失: {train_loss:.4f}")

            self.accelerator.wait_for_everyone()
            eval_metrics = self._evaluate()

            if self.accelerator.is_main_process:
                print(f"验证指标: {eval_metrics}")
                current_eval_score = eval_metrics.get("score_sum", float("-inf"))

                if current_eval_score > self.best_eval_score + self.config.cfg_train.min_delta:
                    print(f"验证分数改善 ({self.best_eval_score:.4f} -> {current_eval_score:.4f})，保存最佳LoRA适配器...")
                    self.best_eval_score = current_eval_score
                    self.epochs_no_improve = 0

                    # 【最终关键修改 - F】: 保存逻辑简化
                    best_model_save_dir = os.path.join(self.output_dir, "best_model_stage2")
                    os.makedirs(best_model_save_dir, exist_ok=True)

                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    
                    # 现在，我们只需要保存LoRA权重。
                    # GNN和Injection模块的权重在推理时从第一阶段的产出中加载。
                    unwrapped_model.save_pretrained(best_model_save_dir)
                    
                    print(f"最佳LoRA适配器已保存到：{best_model_save_dir}")
                else:
                    self.epochs_no_improve += 1
                    print(f"验证分数未改善。连续无改善 Epoch 数: {self.epochs_no_improve}/{self.patience}")

                if self.epochs_no_improve >= self.patience:
                    print(f"早停触发！")
                    self.accelerator.end_training()
                    break
            
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print("\n--- 阶段二训练结束 ---")
            if self.best_eval_score > float("-inf"):
                print(f"最终最佳LoRA适配器（score_sum: {self.best_eval_score:.4f}）已保存到：{os.path.join(self.output_dir, 'best_model_stage2')}")
            else:
                print("训练过程中没有产生有效的最佳模型。")