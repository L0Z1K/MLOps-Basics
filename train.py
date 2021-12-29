import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import CoLAModel


def main():
    cola_data = DataModule()
    cola_model = CoLAModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best.ckpt",
        monitor="valid/loss",
        mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics",
                               entity="sybaek",
                               name="bert")

    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback,
                   early_stopping_callback],
        log_every_n_steps=10,
    )

    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()