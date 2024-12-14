import hydra
from omegaconf import DictConfig
import os
import torch
import pytorch_lightning as pl
from data_module import ProbDistDataModule
from lightning_module import EntropyPredictor

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Инициализируем DataModule
    dm = ProbDistDataModule(
        size=cfg.data.size,
        vector_dim=cfg.data.vector_dim,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    # Инициализируем LightningModule
    model = EntropyPredictor(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=cfg.model.output_dim,
        learning_rate=cfg.model.learning_rate
    )

    # Тренер
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices
    )

    # Обучение и тестирование
    trainer.fit(model, dm)
    trainer.test(model, dm)

    torch.save(model.state_dict(), "./models/model_weights.pt")
    # dvc add models/model_weights.pt
    # dvc commit

if __name__ == "__main__":
    main()
