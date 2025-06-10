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

        for batch_idx, batch in enumerate(progress_bar): # 添加 batch_idx
            batched_graph = batch[0]
            prompt_texts = batch[1]
            label_texts = batch[2]

            # 拼接 prompt + label
            prompt_label_texts = [p + l for p, l in zip(prompt_texts, label_texts)]

            # 编码拼接后的输入（用于 input_ids 和 labels）
            encoded_inputs = self.tokenizer(
                prompt_label_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=1024,
            ).to(self.device)

            input_ids = encoded_inputs["input_ids"]
            attention_mask = encoded_inputs["attention_mask"]

            # 编码 prompt，用于构造 labels 的 mask
            encoded_prompts = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=1024, # 确保这里的max_length与上面一致
                ).to(self.device)

            # 构造 labels：prompt 部分为 -100，label 部分为目标
            labels = input_ids.clone()
            prompt_lengths = (encoded_prompts["attention_mask"] == 1).sum(dim=1)

            

            for i, prompt_len in enumerate(prompt_lengths):
                # 确保 prompt_len 不会超过 labels 的维度
                if prompt_len > labels.shape[1]:
                    print(f"WARNING: prompt_len {prompt_len} exceeds labels.shape[1] {labels.shape[1]} for sample {i}")
                    prompt_len = labels.shape[1] # 截断以避免索引错误
                labels[i, :prompt_len] = -100  # 屏蔽 prompt 部分

            

            # 模型前向传播（含 batched_graph 注入）
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batched_graph=batched_graph, # 确认 batched_graph 类型和形状是否正确
                labels=labels,
            )

            loss = outputs.loss

            

            # 反向传播与优化
            self.accelerator.backward(loss)
            if self.config.cfg_train.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.cfg_train.max_grad_norm
                )

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_dataloader)
        # print(f"\n--- DEBUG: End of Epoch, Average Loss: {avg_loss:.4f} ---") # 打印最终平均损失
        return avg_loss

    def _evaluate(self):
        self.model.eval()
        all_predictions_ids = []
        all_loss = []
        all_label_texts_for_eval = [] # 这是每个进程本地收集的，最后会统一到主进程

        eval_progress_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_data in eval_progress_bar:
            with torch.no_grad():
                batched_graph = None
                prompt_texts = None
                label_texts_for_supervision = None
                is_train_val_mode = len(batch_data) == 3  # train/val 模式

                if is_train_val_mode:
                    batched_graph, prompt_texts, label_texts_for_supervision = batch_data
                else:
                    batched_graph, prompt_texts = batch_data
                batched_graph = batched_graph.to(self.device)

                encoded_inputs = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                input_ids = encoded_inputs["input_ids"]
                attention_mask = encoded_inputs["attention_mask"]

                unwrapped_model = self.accelerator.unwrap_model(self.model)
                h_change, _ = unwrapped_model.multimodal_emotion_gnn(batched_graph)

                generated_ids = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # batched_graph=batched_graph,
                    h_change=h_change,
                    pad_token_id=self.tokenizer.pad_token_id, 
                    max_new_tokens=self.config.cfg_train.max_new_tokens,
                    num_beams=self.config.cfg_train.num_beams,
                    do_sample=self.config.cfg_train.do_sample,
                    temperature=self.config.cfg_train.temperature,
                    top_p=self.config.cfg_train.top_p,
                    top_k=self.config.cfg_train.top_k,
                    repetition_penalty=self.config.cfg_train.repetition_penalty,
                )
                prompt_length = input_ids.shape[1]
                sliced_generated_ids = []
                for i in range(generated_ids.shape[0]):

                    sliced_ids = generated_ids[i, prompt_length:].tolist() # 转换为 Python list
                    sliced_generated_ids.append(sliced_ids)
                

                all_predictions_ids.extend(sliced_generated_ids) # 这是每个进程本地收集的列表

                if is_train_val_mode:
                    # **在此处对 label_texts_for_supervision 进行初步清理，防止空字符串或特殊字符**
                    # 例如，去除前后空白，并确保非空
                    cleaned_label_texts = [
                        txt.strip() if txt and txt.strip() else "[EMPTY_LABEL_PLACEHOLDER]"
                        for txt in label_texts_for_supervision
                    ]
                    all_label_texts_for_eval.extend(cleaned_label_texts)

                    # 损失计算部分保持不变
                    prompt_label_texts = [
                        p + l for p, l in zip(prompt_texts, label_texts_for_supervision)
                    ]
                    encoded = self.tokenizer(
                        prompt_label_texts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=1024,
                    ).to(self.device)
                    full_input_ids = encoded["input_ids"]
                    full_attention_mask = encoded["attention_mask"]
                    encoded_prompt_only = self.tokenizer(
                            prompt_texts,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=1024,
                        ).to(self.device)
                    labels = full_input_ids.clone()
                    prompt_lengths = (encoded_prompt_only["attention_mask"] == 1).sum(dim=1)
                    for i, prompt_len in enumerate(prompt_lengths):
                        labels[i, :prompt_len] = -100
                    outputs = self.model(
                        input_ids=full_input_ids,
                        attention_mask=full_attention_mask,
                        batched_graph=batched_graph,
                        labels=labels,
                    )
                    all_loss.append(outputs["loss"].item())

        metrics = {}
        # **所有 gather_for_metrics 操作都应该在所有进程上执行，然后只在主进程上处理结果**
        # 这是 accelerate 的设计，它会在底层同步和收集
        # all_predictions_ids 是 List[List[int]]
        all_predictions_gathered = self.accelerator.gather_for_metrics(all_predictions_ids)
        
        # all_label_texts_for_eval 是 List[str]
        # all_references_gathered 应该包含所有进程的 label_texts，
        # 即使某个进程的 all_label_texts_for_eval 暂时为空，gather 操作也会正确同步
        if all_label_texts_for_eval: # 仅当有标注数据时才进行指标计算
            all_references_gathered = self.accelerator.gather_for_metrics(all_label_texts_for_eval)
            all_loss_gathered = self.accelerator.gather_for_metrics(all_loss)
            
            if self.accelerator.is_main_process:
                # 在主进程进行解码和指标计算
                decoded_predictions = self.evaluator.decode_generated_tokens(all_predictions_gathered)

                # **在此处添加调试打印和空字符串检查，此时数据是完整的**
                print(f"\n[Rank {self.accelerator.process_index}] --- DEBUG: Formatted Texts for Metrics ---")
                print(f"[Rank {self.accelerator.process_index}] Sample decoded_predictions (first 5): {decoded_predictions[:5]}")
                print(f"[Rank {self.accelerator.process_index}] Sample all_references_gathered (first 5): {all_references_gathered[:5]}")

                empty_decoded_preds = [i for i, s in enumerate(decoded_predictions) if not s.strip()]
                empty_gathered_refs = [i for i, s in enumerate(all_references_gathered) if not s.strip()]
                if empty_decoded_preds:
                    print(f"[Rank {self.accelerator.process_index}] WARNING: decoded_predictions contains empty strings at indices: {empty_decoded_preds}")
                if empty_gathered_refs:
                    print(f"[Rank {self.accelerator.process_index}] WARNING: all_references_gathered contains empty strings at indices: {empty_gathered_refs}")
                print(f"[Rank {self.accelerator.process_index}] --- END DEBUG: Formatted Texts for Metrics ---")


                print(f"Decoded Predictions for a sample (first): {decoded_predictions[0]}") # 打印第一个
                print(f"References for a sample (first): {all_references_gathered[0]}") # 打印第一个

                metrics = self.evaluator.compute_metrics(
                    predictions=decoded_predictions,
                    references=all_references_gathered, # references 是 Dict 列表，Evaluator 内部会处理
                )

                avg_eval_loss = torch.tensor(all_loss_gathered).mean().item()
                metrics["eval_loss"] = round(avg_eval_loss, 4)

        # 无论有没有 is_train_val_mode，最终都应该返回 metrics
        # 确保所有进程都执行到这里，即使只有主进程返回真正的 metrics dict
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
