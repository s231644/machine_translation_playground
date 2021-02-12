# Basic
import numpy as np
import random
from pathlib import Path

# PyTorch
import torch

# PyTorch Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from runner.runner import Runner
from models.seq2seq.convlstm_seq2seq import ConvGRUSeq2Seq as Model
from readers.words_to_digits_reader import Words2DigitsReader as Reader

# set seed
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reader = Reader("data/numerals_s2s/words2digits")

# prepare model
INPUT_DIM = len(reader.SRC.vocab)
OUTPUT_DIM = len(reader.TRG.vocab)
SRC_PAD_IDX = reader.SRC.vocab.stoi[reader.SRC.pad_token]

model = Model(
    input_dim=INPUT_DIM,
    enc_emb_dim=5,
    enc_hid_dim=8,
    enc_pad_ix=SRC_PAD_IDX,
    enc_dropout=0.1,
    output_dim=OUTPUT_DIM,
    dec_emb_dim=4,
    dec_hid_dim=8,
    dec_dropout=0.1,
    device=device
)

model = model.to(device)

runner = Runner(
    model=model,
    reader=reader,
    train_batch_size=64,
    val_batch_size=128
)

logger = TensorBoardLogger(
    Path("./logs"),
    name="gru_seq2seq"
)

trainer = Trainer(
    logger=logger,
    max_epochs=1,
)

print(runner.predict("пять сто шесть -надцать"))
trainer.fit(runner)
runner.model.to(device)
print(runner.predict("пять сто шесть -надцать"))
