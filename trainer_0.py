import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
from accelerate import Accelerator


from MECE_data.mecr_dataset import MECRDataset
from MECE_data.collate_to_graph_batch import CustomCollate
from MECE_data.build_emotion_graph import build_emotion_graph
from model.multimodal_emotion_gnn import MultimodalEmotionGNN
from model.gnn_wraper import ContrastiveWrapper


class TrainerStage0:
    def __init__(self, cfg: dataclass):
        """
        [阶段零] 训练器初始化：用于GNN和融合模块的对比学习预训练。
        """
        # 1. 初始化 Accelerator 和基本配置
        self.accelerator = Accelerator()
        self.config = cfg
        self.device = self.accelerator.device
        print(f"===== STAGE 0: Contrastive Pre-training - DEVICE: {self.device} =====")

        output_dir = os.path.dirname(self.config.model_save_path)
        self.save_path = os.path.join(output_dir, "stage0_gnn_pretrained.pt")
        os.makedirs(output_dir, exist_ok=True)

        # 2. 数据加载 (完全复用您已有的代码逻辑)
        # 注意mode='train'即可，因为你的collate_fn会返回我们需要的所有东西
        train_dataset = MECRDataset(
            json_path=self.config.json_path_train,
            feature_root=self.config.feature_root_train,
            mode="train",
            tokenizer=self.config.tokenizer_name,
            bert_model=self.config.bert_name,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,  # 从配置中读取
            shuffle=True,
            collate_fn=CustomCollate(build_emotion_graph),
            num_workers=self.config.num_workers,
        )

        # 3. 模型初始化
        # 先实例化你自己的GNN模型
        gnn_model_instance = MultimodalEmotionGNN(
            in_dim=self.config.in_dim,
            gnn_dim=self.config.gnn_dim,
            gnn_in_dim=self.config.gnn_in_dim,
            gnn_hidden_dim=self.config.gnn_hidden_dim,
            gnn_out_dim=self.config.gnn_out_dim,
            gnn_heads=self.config.gnn_num_heads,
        )

        # 用ContrastiveWrapper包装起来
        self.model = ContrastiveWrapper(
            gnn_model=gnn_model_instance,
            text_encoder_name=self.config.bert_name,
            projection_dim=self.config.projection_dim,
        )

        # 4. 优化器
        # 冻结文本编码器，只训练GNN和投影层
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.config.learning_rate
        )

        # 5. 使用Accelerator进行准备
        (self.model, self.optimizer, self.train_loader) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch",
            disable=not self.accelerator.is_local_main_process,
        )

        for batched_graph, _, label_texts in progress_bar:  # 我们忽略 prompt_texts
            # 模型forward
            loss = self.model(batched_graph=batched_graph, text_list=label_texts)

            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def train(self):
        print("--- Starting Stage 0: Contrastive Pre-training ---")
        best_loss = float("inf")

        for epoch in range(self.config.epochs):
            if self.accelerator.is_main_process:
                print(f"\n--- STAGE 0: Epoch {epoch + 1}/{self.config.epochs} ---")

            avg_train_loss = self._train_epoch()

            if self.accelerator.is_main_process:
                print(
                    f"Epoch {epoch + 1} 's Average Training Loss: {avg_train_loss:.4f}"
                )

                # 以训练损失作为保存模型的依据
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    print(f"New best loss: {best_loss:.4f}. Saving GNN model...")
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self.accelerator.save(
                        unwrapped_model.graph_encoder.state_dict(), self.save_path
                    )
                    print(f"GNN weights saved to {self.save_path}")

        if self.accelerator.is_main_process:
            print("\n--- Stage 0 Pre-training Finished ---")
            print(f"Best GNN weights saved at: {self.save_path}")
