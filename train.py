import torch
import pytorch_lightning as pl
from model import NN
from datamodule import DataModule
from dataset import Dataset
import config
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from utils import show_images_and_masks

def main():
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v0")

    metadata = pd.read_csv(config.METADATA_CSV_PATH)

    model = NN(learning_rate=config.LEARNING_RATE)

    dm = DataModule(img_path=config.IMAGE_PATH,
                    mask_path=config.MASK_PATH,
                    metadata=metadata,
                    batch_size=config.BATCH_SIZE,
                    num_workers=config.NUM_WORKERS)

    trainer = pl.Trainer(accelerator=config.ACCELERATOR,
                        logger=logger,
                        devices=config.DEVICES,
                        min_epochs=1,
                        max_epochs=config.NUM_EPOCHS,
                        precision=config.PRECISION)

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()