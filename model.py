from typing import Any, Sequence, Union
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
import torchvision
import segmentation_models_pytorch as smp
import numpy as np


class Block(nn.Module):
    def __init__(self, inputs = 3, middles = 64, outs = 64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inputs, middles, 3, 1, 1)
        self.conv2 = nn.Conv2d(middles, outs, 3, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outs)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn(self.conv2(x)))
        return self.pool(x), x


class UNet(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.en1 = Block(3, 64, 64)
        self.en2 = Block(64, 128, 128)
        self.en3 = Block(128, 256, 256)
        self.en4 = Block(256, 512, 512)
        self.en5 = Block(512, 1024, 512)
        
        self.upsample4 = nn.ConvTranspose2d(512, 512, 2, stride = 2)
        self.de4 = Block(1024, 512, 256)
        
        self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride = 2)
        self.de3 = Block(512, 256, 128)
        
        self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride = 2)
        self.de2 = Block(256, 128, 64)
        
        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride = 2)
        self.de1 = Block(128, 64, 64)
        
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1, stride = 1, padding = 0)
        
    def forward(self, x):
        # x: [bs, 3, 256, 256]
        
        x, e1 = self.en1(x)
        # x: [bs, 64, 128, 128]
        # e1: [bs, 64, 256, 256]
        
        x, e2 = self.en2(x)
        # x: [bs, 128, 64, 64]
        # e2: [bs, 128, 128, 128]
        
        x, e3 = self.en3(x)
        # x: [bs, 256, 32, 32]
        # e3: [bs, 256, 64, 64]
        
        x, e4 = self.en4(x)
        # x: [bs, 512, 16, 16]
        # e4: [bs, 512, 32, 32]
        
        _, x = self.en5(x)
        # x: [bs, 512, 16, 16]
        
        
        x = self.upsample4(x)
        # x: [bs, 512, 32, 32]
        x = torch.cat([x, e4], dim=1)
        # x: [bs, 1024, 32, 32]
        _,  x = self.de4(x)
        # x: [bs, 256, 32, 32]
        
        x = self.upsample3(x)
        # x: [bs, 256, 64, 64]
        x = torch.cat([x, e3], dim=1)
        # x: [bs, 512, 64, 64]
        _, x = self.de3(x)
        # x: [bs, 128, 64, 64]
        
        x = self.upsample2(x)
        # x: [bs, 128, 128, 128]
        x = torch.cat([x, e2], dim=1)
        # x: [bs, 256, 128, 128]
        _, x = self.de2(x)
        # x: [bs, 64, 128, 128]
        
        x = self.upsample1(x)
        # x: [bs, 64, 256, 256]
        x = torch.cat([x, e1], dim=1)
        # x: [bs, 128, 256,256, 256
        _, x = self.de1(x)
        # x: [bs, 64, 256, 256]
        
        x = self.conv_last(x)
        # x: [bs, 1, 256, 256]
        
        # x = x.squeeze(1)         
        return x

class NN(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.model = smp.Unet()  # Assuming you are using the UNet model from segmentation_models_pytorch
        self.lr = learning_rate
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        return mask
    
    def shared_step(self, batch, stage):
        image, mask = batch
        mask = mask[:, 0:1, :, :]

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Calculate metrics
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return loss, tp, fp, fn, tn

    def shared_epoch_end(self, stage, losses, tps, fps, fns, tns):
        # Calculate mean loss
        mean_loss = torch.stack(losses).mean()

        # Calculate mean metrics
        mean_tp = torch.cat(tps).mean()
        mean_fp = torch.cat(fps).mean()
        mean_fn = torch.cat(fns).mean()
        mean_tn = torch.cat(tns).mean()

        per_image_iou = smp.metrics.iou_score(mean_tp, mean_fp, mean_fn, mean_tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(mean_tp, mean_fp, mean_fn, mean_tn, reduction="micro")
        recall = smp.metrics.recall(mean_tp, mean_fp, mean_fn, mean_tn, reduction="micro")
        precision = smp.metrics.precision(mean_tp, mean_fp, mean_fn, mean_tn, reduction="micro")
        f1_score = smp.metrics.f1_score(mean_tp, mean_fp, mean_fn, mean_tn, reduction="micro")
        accuracy = smp.metrics.accuracy(mean_tp, mean_fp, mean_fn, mean_tn, reduction="macro")

        metrics = {
            f"{stage}_loss": mean_loss,
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
            f"{stage}_accuracy": accuracy,
            f"{stage}_f1_score": f1_score,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss, tp, fp, fn, tn = self.shared_step(batch, 'train')
        return {'loss': loss, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def validation_step(self, batch, batch_idx):
        loss, tp, fp, fn, tn = self.shared_step(batch, 'val')
        return {'loss': loss, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def test_step(self, batch, batch_idx):
        loss, tp, fp, fn, tn = self.shared_step(batch, 'test')
        return {'loss': loss, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def training_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        tps = [x['tp'] for x in outputs]
        fps = [x['fp'] for x in outputs]
        fns = [x['fn'] for x in outputs]
        tns = [x['tn'] for x in outputs]

        self.shared_epoch_end('train', losses, tps, fps, fns, tns)

    def validation_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        tps = [x['tp'] for x in outputs]
        fps = [x['fp'] for x in outputs]
        fns = [x['fn'] for x in outputs]
        tns = [x['tn'] for x in outputs]

        self.shared_epoch_end('val', losses, tps, fps, fns, tns)

    def test_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        tps = [x['tp'] for x in outputs]
        fps = [x['fp'] for x in outputs]
        fns = [x['fn'] for x in outputs]
        tns = [x['tn'] for x in outputs]

        self.shared_epoch_end('test', losses, tps, fps, fns, tns)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)