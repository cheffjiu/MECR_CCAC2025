import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 关闭tokenizer的并行化，避免与dataloader的多线程冲突，导致死锁


current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path)))
sys.path.extend(
    os.path.join(project_root, sub) for sub in ["", "model", "MECE_data", "utils"]
)
import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoModelForCausalLM,
)
from tqdm import tqdm
from dataclasses import dataclass
from accelerate import Accelerator

# Import PEFT modules
from peft import LoraConfig, get_peft_model, TaskType

# 导入自定义模块
from model.Qwen_with_Injection import QwenWithInjection

# from model.emotion_graph_encoder import EmotionGraphEncoder
from model.Inject_to_llm import InjectionModule
from MECE_data.mecr_dataset import MECRDataset
from MECE_data.collate_to_graph_batch import CustomCollate
from MECE_data.build_emotion_graph import build_emotion_graph

# from model.feature_fusion import CrossModalAttention
from model.multimodal_emotion_gnn import MultimodalEmotionGNN
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
            self.config.cfg_dataset_dataloader.tokenizer_name, trust_remote_code=True
        )

        # 3. 初始化模型
        qwen_base_model = AutoModelForCausalLM.from_pretrained(
            self.config.cfg_model.llm_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        self.model = QwenWithInjection(
            qwen_base_model,
            self.config.cfg_model,
        )

        # --- LoRA Configuration for lm_head ---
        lora_config = LoraConfig(
            r=self.config.cfg_lora.lora_r,
            lora_alpha=self.config.cfg_lora.lora_alpha,
            target_modules=["lm_head"],
            lora_dropout=self.config.cfg_lora.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        # --- End LoRA Configuration ---

        # 4. 数据集和数据加载器
        self.train_dataset = MECRDataset(
            json_path=self.config.cfg_dataset_dataloader.json_path_train,
            feature_root=self.config.cfg_dataset_dataloader.feature_root_trian,
            mode="train",
            tokenizer=self.config.cfg_dataset_dataloader.tokenizer_name,
            bert_model=self.config.cfg_dataset_dataloader.bert_name,
        )
        self.eval_dataset = MECRDataset(
            json_path=self.config.cfg_dataset_dataloader.json_path_val,
            feature_root=self.config.cfg_dataset_dataloader.feature_root_val,
            mode="val",
            tokenizer=self.config.cfg_dataset_dataloader.tokenizer_name,
            bert_model=self.config.cfg_dataset_dataloader.bert_name,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.cfg_dataset_dataloader.batch_size,
            shuffle=True,
            collate_fn=CustomCollate(             
                build_emotion_graph
            ),
            num_workers=self.config.cfg_dataset_dataloader.num_workers,
            pin_memory=True,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.cfg_dataset_dataloader.batch_size,
            shuffle=False,
            collate_fn=CustomCollate(
                build_emotion_graph
            ),
            num_workers=self.config.cfg_dataset_dataloader.num_workers,
            pin_memory=True,
        )

        # 5. 优化器和学习率调度器数
        no_decay = ["bias", "LayerNorm.weight"]
        # 只需收集 self.model (QwenWithInjection，它现在包含了所有子模块) 的参数
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() 
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.cfg_train.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() # <--- 只需要遍历 self.model
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.cfg_train.weight_decay,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.config.cfg_train.learning_rate
        )


        num_training_steps = len(self.train_dataloader) * self.config.cfg_train.num_train_epochs
        num_warmup_steps = int(num_training_steps * self.config.cfg_train.warmup_ratio)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
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

        for batch in progress_bar:

            batched_graph = batch[0]
            prompt_texts = batch[1]
            label_texts = batch[2]

            encoded_inputs = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            ).to(self.device)
            input_ids = encoded_inputs["input_ids"]
            attention_mask = encoded_inputs["attention_mask"]

            encoded_labels = self.tokenizer(
                label_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            ).to(self.device)
            labels = encoded_labels["input_ids"]
            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
            # print(f"Shape of labels after tokenization and masking: {labels.shape}")
            # 模型前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batched_graph=batched_graph,
                labels=labels,
            )
            loss = outputs.loss

            # 反向传播和优化 (使用 accelerator)
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
        return avg_loss

    def _evaluate(self):
        self.model.eval()
        all_predictions_ids = []
        all_loss = []

        eval_progress_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_data in eval_progress_bar:
            with torch.no_grad():
                # 解包 batch_data
                # 在 train/val 模式下：batched_graph, prompt_texts, label_texts
                # 在 test 模式下：batched_graph, prompt_texts
                if len(batch_data) == 3:  # train/val 模式
                    batched_graph, prompt_texts, label_texts = batch_data
                else:  # test 模式
                    batched_graph, prompt_texts = batch_data
                    label_texts = None  # 测试模式没有 label/rationale

                batched_graph = batched_graph.to(self.device)

                # 只有在有 label 的模式下才进行损失计算和生成
                if label_texts is not None:  # 如果是训练/验证模式
                    # 编码 prompt 输入
                    encoded_inputs = self.tokenizer(
                        prompt_texts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    input_ids = encoded_inputs["input_ids"]
                    attention_mask = encoded_inputs["attention_mask"]

                    # 编码 label 文本
                    encoded_labels = self.tokenizer(
                        label_texts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    labels = encoded_labels["input_ids"]
                    labels = torch.where(
                        labels == self.tokenizer.pad_token_id, -100, labels
                    )
                    # 计算损失
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        batched_graph=batched_graph,
                        labels=labels,
                    )
                    all_loss.append(outputs["loss"].item())

                    # 解除accelerator的包装,避免分布式问题
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    # 生成预测tokens_ids
                    generated_ids = unwrapped_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        batched_graph=batched_graph, # <-- GNN 输入，传递给 QwenWithInjection.forward
                        pad_token_id=self.tokenizer.eos_token_id, # 通常设为 EOS token ID

                        # === 从配置中读取并应用的生成参数 ===
                        max_new_tokens=self.config.cfg_train.max_new_tokens,
                        num_beams=self.config.cfg_train.num_beams,
                        do_sample=self.config.cfg_train.do_sample,
                        temperature=self.config.cfg_train.temperature,
                        top_p=self.config.cfg_train.top_p,
                        top_k=self.config.cfg_train.top_k,
                        repetition_penalty=self.config.cfg_train.repetition_penalty,
                        # ==================================
                    )

                    all_predictions_ids.extend(generated_ids.cpu())

                else:  # 测试模式只生成预测
                    # 编码 prompt 输入
                    encoded_inputs = self.tokenizer(
                        prompt_texts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    input_ids = encoded_inputs["input_ids"]
                    attention_mask = encoded_inputs["attention_mask"]
                    # 解除accelerator的包装,避免分布式问题
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    # 生成预测tokens_ids
                    generated_ids = unwrapped_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        batched_graph=batched_graph, 
                        pad_token_id=self.tokenizer.eos_token_id, # 通常设为 EOS token ID

                        # === 从配置中读取并应用的生成参数 ===
                        max_new_tokens=self.config.cfg_train.max_new_tokens,
                        num_beams=self.config.cfg_train.num_beams,
                        do_sample=self.config.cfg_train.do_sample,
                        temperature=self.config.cfg_train.temperature,
                        top_p=self.config.cfg_train.top_p,
                        top_k=self.config.cfg_train.top_k,
                        repetition_penalty=self.config.cfg_train.repetition_penalty,
                        
                    )
                    all_predictions_ids.extend(generated_ids.cpu())

        # 在所有进程上收集结果
        all_predictions_gathered = self.accelerator.gather_for_metrics(
            all_predictions_ids
        )

        metrics = {}
        # 只有在 train/val 模式下计算这些指标
        if label_texts is not None:  # 如果是训练/验证模式
            all_references_gathered = self.accelerator.gather_for_metrics(label_texts)
            all_loss_gathered = self.accelerator.gather_for_metrics(all_loss)

            # 在主进程上计算指标
            if self.accelerator.is_main_process:
                decoded_predictions = self.tokenizer.batch_decode(
                    all_predictions_gathered, skip_special_tokens=True
                )
                metrics = self.evaluator.compute_metrics(
                    predictions=decoded_predictions,
                    references=all_references_gathered,
                )
                avg_eval_loss = torch.cat(all_loss_gathered).mean().item()
                metrics["eval_loss"] = round(avg_eval_loss, 4)

            return metrics
        else:  # 测试模式，不计算损失和复杂指标
            if self.accelerator.is_main_process:
                decoded_predictions = self.tokenizer.batch_decode(
                    all_predictions_gathered, skip_special_tokens=True
                )
                metrics["generated_texts"] = decoded_predictions
            return metrics

    def train(self):
        
        start_validation_epoch = self.config.cfg_train.start_validation_epoch # 从配置中获取起始验证 epoch

        for epoch in range(self.config.cfg_train.num_train_epochs):
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch + 1}/{self.config.cfg_train.num_train_epochs}")

            train_loss = self._train_epoch()
            if self.accelerator.is_main_process:
                print(f"训练损失: {train_loss:.4f}")

            # --- 条件判断：从指定 epoch 开始进行验证和保存 ---
            if (epoch + 1) >= start_validation_epoch:
                # 每个 Epoch 结束时保存一次模型 (主进程)
                if self.accelerator.is_main_process:  
                    epoch_save_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1}")
                    os.makedirs(epoch_save_dir, exist_ok=True)

                    # 1. 解包所有经过 accelerator.prepare 包装过的主模型
                    unwrapped_model = self.accelerator.unwrap_model(self.model)

                    # 2. 保存 LoRA 适配器 (QwenWithInjection 是 PeftModel 的实例)
                    print(f"保存 LoRA 适配器到 {epoch_save_dir}/lora_adapters...")
                    unwrapped_model.save_pretrained(os.path.join(epoch_save_dir, "lora_adapters"))

                    # 3. 保存 InjectionModule 的参数 (通过主模型访问其子模块)
                    print(
                        f"保存 InjectionModule 参数到 {epoch_save_dir}/injection_module.pt..."
                    )
                    torch.save(
                        unwrapped_model.injection_module.state_dict(),
                        os.path.join(epoch_save_dir, "injection_module.pt"),
                    )

                    # 4. 保存 MultimodalEmotionGNN 的参数 (通过主模型访问其子模块)
                    print(f"保存 MultimodalEmotionGNN 参数到 {epoch_save_dir}/multimodal_emotion_gnn.pt...")
                    torch.save(
                        unwrapped_model.multimodal_emotion_gnn.state_dict(),
                        os.path.join(epoch_save_dir, "multimodal_emotion_gnn.pt"),
                    )
                    print(f"模型组件已保存到 {epoch_save_dir}")

                # **等待所有进程同步，然后进行评估**
                self.accelerator.wait_for_everyone() 

                eval_metrics = self._evaluate()
                
                # 只有主进程处理评估结果和早停逻辑
                if self.accelerator.is_main_process:
                    print(f"验证指标: {eval_metrics}")

                    current_eval_score = eval_metrics.get("score_sum", float("-inf")) # 假设 score_sum 是你的主要评估指标

                    if current_eval_score > self.best_eval_score + self.min_delta:
                        print(
                            f"验证分数改善 ({self.best_eval_score:.4f} -> {current_eval_score:.4f})，保存最佳模型..."
                        )
                        self.best_eval_score = current_eval_score
                        self.epochs_no_improve = 0

                        # 最佳模型保存 (只保存最佳的 LoRA, InjectionModule 和 GNN 模块)
                        best_model_save_dir = os.path.join(self.output_dir, "best_model")
                        os.makedirs(best_model_save_dir, exist_ok=True)

                        # 再次解包确保获取当前最佳模型的未包装版本
                        unwrapped_model = self.accelerator.unwrap_model(self.model)

                        # 1. 保存最佳 LoRA 适配器
                        print(f"保存最佳 LoRA 适配器到 {best_model_save_dir}/lora_adapters...")
                        unwrapped_model.save_pretrained(os.path.join(best_model_save_dir, "lora_adapters"))

                        # 2. 保存最佳 InjectionModule 的参数
                        print(
                            f"保存最佳 InjectionModule 参数到 {best_model_save_dir}/injection_module.pt..."
                        )
                        torch.save(
                            unwrapped_model.injection_module.state_dict(),
                            os.path.join(best_model_save_dir, "injection_module.pt"),
                        )

                        # 3. 保存最佳 MultimodalEmotionGNN 的参数
                        print(
                            f"保存最佳 MultimodalEmotionGNN 参数到 {best_model_save_dir}/multimodal_emotion_gnn.pt..."
                        )
                        torch.save(
                            unwrapped_model.multimodal_emotion_gnn.state_dict(),
                            os.path.join(best_model_save_dir, "multimodal_emotion_gnn.pt"),
                        )
                        print(f"最佳模型组件已保存到：{best_model_save_dir}")

                    else:
                        self.epochs_no_improve += 1
                        print(
                            f"验证分数未改善。连续无改善 Epoch 数: {self.epochs_no_improve}/{self.patience}"
                        )

                    if self.epochs_no_improve >= self.patience:
                        print(f"早停触发！连续 {self.patience} 个 Epoch 未见改善。")
                        break
            else:
                # 如果不进行验证和保存，确保在每个 epoch 结束时所有进程都同步
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch + 1}：跳过验证和模型保存。")

            # 在每个 epoch 结束时等待所有进程同步，即使不进行验证
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print("训练结束。")
            print(
                f"最终最佳模型所有组件已保存到：{os.path.join(self.output_dir, 'best_model')}"
            )