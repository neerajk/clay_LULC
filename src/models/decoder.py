import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassConfusionMatrix

class EpochSummaryCallback(Callback):
    """Prints a clean summary at the end of every epoch to track progress easily."""
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        train_miou = trainer.callback_metrics.get('train_mIoU')
        if train_loss is not None and train_miou is not None:
            print(f"\n📈 Epoch {trainer.current_epoch} [TRAIN] - Loss: {train_loss:.4f} | mIoU: {train_miou:.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_miou = trainer.callback_metrics.get('val_mIoU')
        if val_loss is not None and val_miou is not None:
            print(f"📉 Epoch {trainer.current_epoch} [VALID] - Loss: {val_loss:.4f} | mIoU: {val_miou:.4f}")
            print("-" * 60)

class CNNUpsamplerDecoder(nn.Module):
    def __init__(self, in_channels=1024, num_classes=20):
        super().__init__()
        print(f"🏗️  Building CNNUpsamplerDecoder (In: {in_channels}, Out: {num_classes})...")
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    def forward(self, x): 
        return self.decoder(x)

class LULCSegmentationModule(pl.LightningModule):
    def __init__(self, num_classes=20, lr=1e-3, ignore_index=0):
        super().__init__()
        self.save_hyperparameters()
        print(f"🤖 Initializing Lightning Module (LR: {lr}, Ignore Index: {ignore_index})...")
        self.decoder = CNNUpsamplerDecoder(in_channels=1024, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        self.train_miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index)
        self.val_miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
        
    def forward(self, x): 
        return self.decoder(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_mIoU', self.train_miou(torch.argmax(logits, dim=1), y), on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_mIoU', self.val_miou(preds, y), on_epoch=True, on_step=False)
        self.val_confmat.update(preds, y)

    def on_validation_epoch_end(self):
        cm = self.val_confmat.compute().cpu().numpy()
        self.last_val_cm = cm 
        self.val_confmat.reset()

    def configure_optimizers(self):
        print(f"🔧 Configuring AdamW Optimizer with ReduceLROnPlateau Scheduler...")
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}