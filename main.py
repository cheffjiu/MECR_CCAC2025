from config import *
from trainer_1 import TrainerStage1
from trainer_2  import TrainerStage2

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
    # trainer = TrainerStage1(cfg)
    trainer=TrainerStage1(cfg)
    trainer.train()
