# Basic
import numpy as np

# PyTorch
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

# PyTorch Lightning
import pytorch_lightning as pl


class Runner(pl.LightningModule):
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 train_iterator,
                 val_iterator,
                 trg_pad_idx,
                 clip,
                 epoch_size,
                 batch_size,
                 with_pad,
                 ):
        super(Runner, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.clip = clip
        self.epoch_size = epoch_size
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

        self.with_pad = with_pad

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        if self.with_pad:
            src, src_len = batch.src
            trg = batch.trg

            output = self.model(src, src_len, trg)
        else:
            src = batch.src
            trg = batch.trg

            output = self.model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = self.criterion(output, trg)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.logger.experiment.add_scalar('train_loss', loss, batch_idx + self.current_epoch * self.epoch_size)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if self.with_pad:
            src, src_len = batch.src
            trg = batch.trg

            output = self.model(src, src_len, trg, 0)
        else:
            src = batch.src
            trg = batch.trg

            output = self.model(src, trg, 0)

        output_dim = output.shape[-1]

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        cost = self.criterion(output, trg)

        self.logger.experiment.add_scalar('cost', cost, batch_idx + self.current_epoch * self.epoch_size)
        return {'val_loss': cost}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters())
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_iterator

    def val_dataloader(self):
        return self.val_iterator
