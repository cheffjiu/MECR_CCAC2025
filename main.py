from config import *
from trainer import Trainer

if __name__ == "__main__":
    cfg_feature_fusion_model = config_feature_fusion_model()
    cfg_emotion_graph_model = config_emotion_graph_model()
    cfg_injection_module = config_injection_module()
    cfg_qwen_llm = config_qwen_llm()
    cfg_dataset_dataloader = config_dataset_dataloader()
    cfg_train = config_train()
    cfg_lora = config_lora()
    cfg = config(
        cfg_feature_fusion_model,
        cfg_emotion_graph_model,
        cfg_injection_module,
        cfg_qwen_llm,
        cfg_dataset_dataloader,
        cfg_train,
        cfg_lora,
    )
    trainer = Trainer(cfg)
    trainer.train()
