import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import CoLAModel


def main():
    cola_data = DataModule()
    cola_model = CoLAModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()