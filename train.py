import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import NN
from datamodule import DataModule

import random
import pandas as pd
from matplotlib import pyplot as plt

import config

import torch
import numpy as np


def main():
    logger = TensorBoardLogger("tb_logs", name="forest segmentation")
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_IoU',
        mode='max',
        save_top_k=1,
        dirpath='checkpoints/',
        filename='best_model_{epoch:02d}-{validation_IoU:.4f}'
    )

    metadata = pd.read_csv(config.METADATA_CSV_PATH)

    model = NN()

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
                         precision=config.PRECISION,
                         callbacks=[checkpoint_callback]
                         )

    trainer.fit(model, dm)
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    trainer.validate(model, dm)
    trainer.test(model, dm)

    dm.setup('train')
    test_dataloader = dm.train_dataloader()

    model.eval()

    random_batch_index = random.randint(0, len(test_dataloader) - 1)

    for batch_idx, batch in enumerate(test_dataloader):

        if batch_idx != random_batch_index:
            continue

        number_of_samples = 3
        images, masks = batch
        with torch.no_grad():
            predictions = model(images)

        fig, axes = plt.subplots(number_of_samples, 4, figsize=(8, 3 * number_of_samples))

        cols = ['Image', 'Real Mask', 'Predicted Mask', 'Overlap']
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for i in range(number_of_samples):
            original_image = (images[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            original_mask = (masks[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            prediction_image = ((predictions[i].sigmoid() > 0.5).numpy().transpose(1, 2, 0) * 255).astype(
                np.uint8)

            axes[i, 0].imshow(original_image)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(original_mask, cmap='gray')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(prediction_image, cmap='gray', vmin=0, vmax=255)
            axes[i, 2].axis('off')

            alpha = 0.5
            blended_image = (original_image * (1.0 - alpha) + prediction_image * alpha).astype('uint8')
            axes[i, 3].imshow(blended_image)
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.show()
        break


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
