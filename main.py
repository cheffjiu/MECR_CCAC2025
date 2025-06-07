from config import *
from trainer import Trainer

if __name__ == "__main__":
    cfg_model=config_model()
    cfg_dataset_dataloader = config_dataset_dataloader()
    cfg_train = config_train()
    cfg_lora = config_lora()
    cfg = config(
        cfg_model,
        cfg_dataset_dataloader,
        cfg_train,
        cfg_lora,
    )
    trainer = Trainer(cfg)
    trainer.train()
