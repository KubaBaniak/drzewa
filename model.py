import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class NN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1, activation=None)
        self.criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.save_hyperparameters()

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage):
        image, mask = batch
        mask = mask.squeeze()

        logits_mask = self.forward(image).squeeze()
        loss = self.criterion(logits_mask, mask)

        pred_mask = (logits_mask.sigmoid() > 0.5).long()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(),
                                               mask.long(),
                                               mode="binary")

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro')
        self.log(f"{stage}_IoU", iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_lss", loss)
        self.log(f"{stage}_accuracy", accuracy)

        return {"loss": loss, "iou": iou}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'validation')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-03)
