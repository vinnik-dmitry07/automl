import numpy as np
import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn.functional import pad
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


class Model(LightningModule):
    def __init__(self, dataset, batch_size, learning_rate):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 39)
        for param in self.bert.parameters():
            param.requires_grad = False

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

    def forward(self, ids, mask):
        outputs = self.bert(input_ids=ids, attention_mask=mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_nb):
        ids, mask, labels = batch
        loss = self.criterion(self(ids, mask), labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask, labels = batch
        labels_hat = self(ids, mask)
        self.log('val_loss', self.criterion(labels_hat, labels), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_matt', self.matt(labels_hat, labels), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.f1(labels_hat, labels), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_macro', self.f1_macro(labels_hat, labels), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.acc(labels_hat, labels), on_epoch=True, prog_bar=True, logger=True)
        # self.logger.experiment.add_pr_curve(
        #   'val_pr', labels, labels_hat[torch.arange(labels_hat.size(0)), labels], self.global_step)

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

    npz = np.load('data_text.npz', allow_pickle=True)
    # lens = list(map(len, npz['x']))
    # pad_len = int(np.percentile(lens, 95.45))
    pad_len = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = [tokenizer(xi, return_tensors='pt')['input_ids'][0] for xi in tqdm(npz['x'])]
    input_ids = [ids[:pad_len] for ids in input_ids]
    input_ids = [
        pad(input=ids, pad=[0, pad_len - len(ids)], value=tokenizer.pad_token_id)
        for ids in input_ids
    ]
    attention_mask = [ids == tokenizer.pad_token_id for ids in input_ids]

    dataset_ = TensorDataset(
        torch.stack(input_ids),
        torch.stack(attention_mask),
        torch.from_numpy(npz['y']).long()
    )

    model = Model(
        dataset=dataset_,
        batch_size=128,
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
        log_every_n_steps=model.batch_size,
    )

    trainer.fit(model)
