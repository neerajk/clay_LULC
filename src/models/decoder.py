import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix
from pytorch_lightning.callbacks import Callback


class EpochSummaryCallback(Callback):
    """Kept for backward compatibility with the training script."""

    def on_validation_epoch_end(self, trainer, pl_module):
        return


class CNNUpsamplerDecoder(nn.Module):
    """
    Dual-head decoder:
    - coarse head predicts 16x16 class logits aligned with CLAY patch tokens
    - full head upsamples to 256x256, seeded by coarse logits
    """

    def __init__(self, in_channels=1024, num_classes=20):
        super().__init__()
        self.coarse_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        self.full_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        coarse_logits = self.coarse_head(x)  # [B, C, 16, 16]
        full_logits = self.full_decoder(x)   # [B, C, 256, 256]
        coarse_upsampled = F.interpolate(coarse_logits, size=full_logits.shape[-2:], mode="nearest")
        full_logits = full_logits + coarse_upsampled
        return full_logits, coarse_logits


class LULCSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        num_classes=20,
        lr=1e-3,
        ignore_index=0,
        class_weights=None,
        weight_decay=1e-4,
        lr_factor=0.5,
        lr_patience=4,
        min_lr=1e-6,
        dice_weight=0.25,
        patch_size=16,
        full_loss_weight=0.6,
        coarse_loss_weight=0.4,
        monitor_metric="val_mIoU_patch",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.decoder = CNNUpsamplerDecoder(in_channels=1024, num_classes=num_classes)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        self.dice_weight = float(dice_weight)

        self.train_confmat = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
        self.train_confmat_patch = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
        self.val_confmat_patch = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)

        self.last_train_loss = None
        self.last_train_ce_loss = None
        self.last_train_dice_loss = None
        self.last_train_miou = None
        self.last_train_miou_patch = None

        self.last_val_loss = None
        self.last_val_ce_loss = None
        self.last_val_dice_loss = None
        self.last_val_miou = None
        self.last_val_miou_patch = None

        self._reset_running_stats()

    def _reset_running_stats(self):
        self._train_stats = {"n": 0, "loss": 0.0, "ce": 0.0, "dice": 0.0}
        self._val_stats = {"n": 0, "loss": 0.0, "ce": 0.0, "dice": 0.0}

    def forward(self, x):
        return self.decoder(x)

    def _soft_dice_loss(self, logits, target, eps=1e-6):
        probs = torch.softmax(logits, dim=1)
        valid_mask = (target != self.hparams.ignore_index).float()
        target_safe = torch.where(target == self.hparams.ignore_index, torch.zeros_like(target), target)
        onehot = torch.nn.functional.one_hot(target_safe, num_classes=self.hparams.num_classes).permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        onehot = onehot * valid_mask

        probs = probs[:, 1:, :, :]
        onehot = onehot[:, 1:, :, :]

        intersection = (probs * onehot).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + eps) / (denom + eps)
        return 1.0 - dice.mean()

    def _downsample_target_mode(self, target):
        # Majority-vote pool: 256x256 -> 16x16 for patch-supervision.
        b, h, w = target.shape
        ps = int(self.hparams.patch_size)
        if h % ps != 0 or w % ps != 0:
            raise ValueError(f"Target size {(h, w)} is not divisible by patch_size={ps}")

        num_classes = int(self.hparams.num_classes)
        onehot = torch.nn.functional.one_hot(
            target.clamp(min=0, max=num_classes - 1),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        gh, gw = h // ps, w // ps
        counts = onehot.reshape(b, num_classes, gh, ps, gw, ps).sum(dim=(3, 5))

        if 0 <= self.hparams.ignore_index < num_classes:
            ignore_counts = counts[:, self.hparams.ignore_index, :, :]
            valid_counts = counts.sum(dim=1) - ignore_counts
            counts[:, self.hparams.ignore_index, :, :] = torch.where(
                valid_counts > 0,
                torch.zeros_like(ignore_counts),
                ignore_counts,
            )
        return counts.argmax(dim=1).long()

    def _miou_from_confmat(self, confmat):
        confmat = confmat.to(torch.float64)
        if torch.any(confmat < 0):
            print("⚠️ [METRIC] Confusion matrix has negative values; clipping to zero.")
            confmat = torch.clamp(confmat, min=0.0)

        tp = confmat.diag()
        fp = confmat.sum(dim=0) - tp
        fn = confmat.sum(dim=1) - tp
        denom = tp + fp + fn

        iou = torch.full_like(tp, float("nan"))
        valid = denom > 0
        iou[valid] = tp[valid] / denom[valid]
        iou[valid] = torch.clamp(iou[valid], min=0.0, max=1.0)

        if 0 <= self.hparams.ignore_index < len(iou):
            iou[self.hparams.ignore_index] = float("nan")

        valid_iou = iou[torch.isfinite(iou)]
        if valid_iou.numel() == 0:
            return torch.tensor(0.0, device=confmat.device), iou
        return valid_iou.mean(), iou

    def on_train_epoch_start(self):
        self._train_stats = {"n": 0, "loss": 0.0, "ce": 0.0, "dice": 0.0}

    def on_validation_epoch_start(self):
        self._val_stats = {"n": 0, "loss": 0.0, "ce": 0.0, "dice": 0.0}

    def training_step(self, batch, batch_idx):
        x, y = batch
        if batch_idx == 0 and self.current_epoch == 0:
            print(f"\n📏 SHAPE CHECK: Embeddings {x.shape} -> Mask {y.shape}")

        full_logits, coarse_logits = self(x)
        y_patch = self._downsample_target_mode(y)

        full_ce = self.criterion(full_logits, y)
        coarse_ce = self.criterion(coarse_logits, y_patch)
        ce_loss = self.hparams.full_loss_weight * full_ce + self.hparams.coarse_loss_weight * coarse_ce

        full_dice = self._soft_dice_loss(full_logits, y)
        coarse_dice = self._soft_dice_loss(coarse_logits, y_patch)
        dice_loss = self.hparams.full_loss_weight * full_dice + self.hparams.coarse_loss_weight * coarse_dice

        loss = ce_loss + (self.dice_weight * dice_loss)

        preds_full = torch.argmax(full_logits, dim=1)
        preds_patch = torch.argmax(coarse_logits, dim=1)
        self.train_confmat.update(preds_full, y)
        self.train_confmat_patch.update(preds_patch, y_patch)

        bsz = x.shape[0]
        self._train_stats["n"] += bsz
        self._train_stats["loss"] += float(loss.detach().cpu()) * bsz
        self._train_stats["ce"] += float(ce_loss.detach().cpu()) * bsz
        self._train_stats["dice"] += float(dice_loss.detach().cpu()) * bsz

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_dice_loss", dice_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        full_logits, coarse_logits = self(x)
        y_patch = self._downsample_target_mode(y)

        full_ce = self.criterion(full_logits, y)
        coarse_ce = self.criterion(coarse_logits, y_patch)
        ce_loss = self.hparams.full_loss_weight * full_ce + self.hparams.coarse_loss_weight * coarse_ce

        full_dice = self._soft_dice_loss(full_logits, y)
        coarse_dice = self._soft_dice_loss(coarse_logits, y_patch)
        dice_loss = self.hparams.full_loss_weight * full_dice + self.hparams.coarse_loss_weight * coarse_dice

        loss = ce_loss + (self.dice_weight * dice_loss)

        preds_full = torch.argmax(full_logits, dim=1)
        preds_patch = torch.argmax(coarse_logits, dim=1)
        self.val_confmat.update(preds_full, y)
        self.val_confmat_patch.update(preds_patch, y_patch)

        bsz = x.shape[0]
        self._val_stats["n"] += bsz
        self._val_stats["loss"] += float(loss.detach().cpu()) * bsz
        self._val_stats["ce"] += float(ce_loss.detach().cpu()) * bsz
        self._val_stats["dice"] += float(dice_loss.detach().cpu()) * bsz

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_dice_loss", dice_loss, on_step=False, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self):
        return

    def on_validation_epoch_end(self):
        train_miou, _ = self._miou_from_confmat(self.train_confmat.compute())
        val_miou, class_iou = self._miou_from_confmat(self.val_confmat.compute())
        train_miou_patch, _ = self._miou_from_confmat(self.train_confmat_patch.compute())
        val_miou_patch, _ = self._miou_from_confmat(self.val_confmat_patch.compute())

        self.train_confmat.reset()
        self.train_confmat_patch.reset()
        self.val_confmat.reset()
        self.val_confmat_patch.reset()

        n_train = max(1, self._train_stats["n"])
        self.last_train_loss = self._train_stats["loss"] / n_train
        self.last_train_ce_loss = self._train_stats["ce"] / n_train
        self.last_train_dice_loss = self._train_stats["dice"] / n_train
        self.last_train_miou = float(train_miou.detach().cpu())
        self.last_train_miou_patch = float(train_miou_patch.detach().cpu())

        n = max(1, self._val_stats["n"])
        self.last_val_loss = self._val_stats["loss"] / n
        self.last_val_ce_loss = self._val_stats["ce"] / n
        self.last_val_dice_loss = self._val_stats["dice"] / n
        self.last_val_miou = float(val_miou.detach().cpu())
        self.last_val_miou_patch = float(val_miou_patch.detach().cpu())

        self.log("train_mIoU", train_miou, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train_mIoU_patch", train_miou_patch, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val_mIoU", val_miou, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_mIoU_patch", val_miou_patch, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        print(f"\n{'=' * 72}")
        print(f"🏁 EPOCH {self.current_epoch} SUMMARY")
        print("-" * 72)
        print(
            f"✅ [TRAIN] Loss: {self.last_train_loss:.4f} | CE: {self.last_train_ce_loss:.4f} | "
            f"Dice: {self.last_train_dice_loss:.4f} | mIoU(full): {self.last_train_miou:.4f} | "
            f"mIoU(patch): {self.last_train_miou_patch:.4f}"
        )
        print(
            f"📊 [VALID] Loss: {self.last_val_loss:.4f} | CE: {self.last_val_ce_loss:.4f} | "
            f"Dice: {self.last_val_dice_loss:.4f} | mIoU(full): {self.last_val_miou:.4f} | "
            f"mIoU(patch): {self.last_val_miou_patch:.4f}"
        )
        if self.trainer and self.trainer.optimizers:
            print(f"🔧 [OPT] LR: {self.trainer.optimizers[0].param_groups[0]['lr']:.7f}")

        for cb in self.trainer.callbacks:
            if cb.__class__.__name__ == "ModelCheckpoint":
                best_path = getattr(cb, "best_model_path", "")
                best_score = getattr(cb, "best_model_score", None)
                if best_path:
                    if best_score is not None:
                        print(f"💾 [CKPT] Best: {best_path} | best={float(best_score):.4f}")
                    else:
                        print(f"💾 [CKPT] Best: {best_path}")
                break

        finite_idx = torch.where(torch.isfinite(class_iou))[0]
        if finite_idx.numel() > 0:
            pairs = [(int(i), float(class_iou[i])) for i in finite_idx]
            pairs_sorted = sorted(pairs, key=lambda x: x[1])
            worst3 = ", ".join([f"{cid}:{score:.3f}" for cid, score in pairs_sorted[:3]])
            best3 = ", ".join([f"{cid}:{score:.3f}" for cid, score in pairs_sorted[-3:]])
            print(f"🧭 [CLASS_IOU full] worst={worst3} | best={best3} | patch_mIoU={self.last_val_miou_patch:.3f}")
        print("=" * 72 + "\n")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.hparams.monitor_metric},
        }
