# PyTorch
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Iterator

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

from metrics.exact_match import ExactMatch
from metrics.bleu import BLEU


class Runner(pl.LightningModule):
    def __init__(
            self,
            model,
            reader,
            train_batch_size: int,
            val_batch_size: int,
    ):
        super(Runner, self).__init__()
        self.model = model
        self.reader = reader
        self.train_dataset = reader.train_ds
        self.val_dataset = reader.valid_ds

        self.train_iterator = Iterator(self.train_dataset, train_batch_size)
        self.val_iterator = Iterator(self.val_dataset, val_batch_size)

        self.dec_pad_token = reader.TRG.pad_token
        self.dec_init_token = reader.TRG.init_token
        self.dec_eos_token = reader.TRG.eos_token

        self.dec_pad_idx = reader.TRG.vocab.stoi[self.dec_pad_token]
        self.dec_init_idx = reader.TRG.vocab.stoi[self.dec_init_token]
        self.dec_eos_idx = reader.TRG.vocab.stoi[self.dec_eos_token]

        self.dec_max_len = reader.TRG.fix_length

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dec_pad_idx)

        self.val_accuracy = Accuracy()
        self.val_exact_match = ExactMatch(ignore_index=self.dec_pad_idx)
        self.val_bleu = BLEU(ignore_index=self.dec_pad_idx)

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def predict(self, x):
        src = self.reader.encode_src(x)
        trg_preds = self.model.predict(
            src,
            init_idx=self.dec_init_idx,
            max_trg_len=self.dec_max_len
        )
        trg_preds = trg_preds[1:]
        preds = self.reader.decode_trg(trg_preds)
        return preds

    def training_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        output = self.model(src, trg, teacher_forcing_ratio=0.5)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = self.criterion(output, trg)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'log': {'tr_loss': loss}}

    def validation_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        output = self.model(src, trg, teacher_forcing_ratio=0)

        trg_preds = output.argmax(2)
        self.val_accuracy.update(trg_preds, trg)
        self.val_exact_match.update(trg_preds, trg)
        self.val_bleu.update(trg_preds, trg)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = self.criterion(output, trg)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_accuracy = self.val_accuracy.compute()
        # avg_exact_match = self.val_exact_match.compute()
        # avg_bleu = self.val_bleu.compute()
        tensorboard_logs = {
            'val_loss': avg_loss,
            # 'val_accuracy': avg_accuracy,
            # 'val_exact_match': avg_exact_match,
            # 'val_bleu': avg_bleu
        }

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters())
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=4, verbose=True
        )
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
