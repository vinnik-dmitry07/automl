import numpy as np
import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Model(LightningModule):
    def __init__(self, dataset, batch_size, learning_rate):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.layers = nn.Sequential(
            nn.Linear(3507, 39),
            nn.Softmax(-1),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.matt = torchmetrics.MatthewsCorrCoef(task='multiclass', num_classes=39)
        self.f1 = torchmetrics.F1Score(task='multiclass')
        self.f1_macro = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=39)
        self.acc = torchmetrics.Accuracy(task='multiclass')
        self.prec = torchmetrics.Precision(task='multiclass')
        self.recall = torchmetrics.Recall(task='multiclass')

        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        self.train_subset, self.val_subset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log('val_loss', self.criterion(y_hat, y), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_matt', self.matt(y_hat, y), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.f1(y_hat, y), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_macro', self.f1_macro(y_hat, y), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.acc(y_hat, y), on_epoch=True, prog_bar=True, logger=True)
        # self.logger.experiment.add_pr_curve('val_pr', y, y_hat[torch.arange(y_hat.size(0)), y], self.global_step)

    # noinspection PyUnresolvedReferences
    def on_train_epoch_end(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
            if params.grad is not None:
                self.logger.experiment.add_histogram(name + '/grad', params.grad, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, amsgrad=True)

    def train_dataloader(self):
        return DataLoader(self.train_subset, num_workers=0, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_subset, num_workers=0, batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x)


if __name__ == '__main__':
    seed_everything(42, workers=True)

    npz = np.load('data_matrix.npz')
    dataset_ = TensorDataset(
        torch.from_numpy(npz['x']),
        torch.from_numpy(npz['y']).long()
    )

    model = Model(
        dataset=dataset_,
        batch_size=256,
        learning_rate=1e-4,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1_macro',
        mode='max',
        save_top_k=3,
        save_last=True,
    )

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=5000,
        precision=32,
        deterministic=True,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)
