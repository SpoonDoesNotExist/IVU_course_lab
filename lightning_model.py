
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import pytorch_lightning as pl
import torch.nn as nn
import torch


class LitClassifier(pl.LightningModule):
    def __init__(self, model, optimizer_class, opt_conf, num_classes):
        super().__init__()
        self.model = model
        self.optimizer_class = optimizer_class
        self.opt_conf = opt_conf
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.precision = MulticlassPrecision(
            num_classes=num_classes, average='macro')
        self.recall = MulticlassRecall(
            num_classes=num_classes, average='macro')
        self.f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        with torch.no_grad():
            self.log("train_loss", loss.item())

            total_norm = sum(
                p.data.norm(2).item() ** 2 for p in self.parameters() if p.requires_grad
            ) ** 0.5
            self.log("train_weight_norm", total_norm)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)

        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), **self.opt_conf)
