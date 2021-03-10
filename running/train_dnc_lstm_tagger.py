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

from runner.runner_tagger import Runner
# from models.pos_tagging.gru_tagging import GRUTagger as Model
# from models.pos_tagging.lstm_tagging import LSTMTagger as Model
# from models.pos_tagging.conv_lstm_encoder import ConvLSTMTagger as Model
from models.pos_tagging.dnc_lstm_tagging import DNCLSTMTagger as Model


from readers.conll2003_reader import CoNLL2003Reader as Reader

# set seed
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reader = Reader("data/en-ud-v2/en-ud-tag.v2")

# prepare model
INPUT_DIM = len(reader.SRC.vocab)
OUTPUT_DIM = len(reader.TRG.vocab)
SRC_PAD_IDX = reader.SRC.vocab.stoi[reader.SRC.pad_token]

model = Model(
    input_dim=INPUT_DIM,
    enc_emb_dim=128,
    enc_hid_dim=128,
    enc_pad_ix=SRC_PAD_IDX,
    enc_dropout=0.1,
    output_dim=OUTPUT_DIM,
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
    Path("./logs/tagging"),
    name="dnclstm_tagger"
)

trainer = Trainer(
    logger=logger,
    max_epochs=100,
)

print(runner.predict("This room is awesome !"))
trainer.fit(runner)
runner.model.to(device)
print(runner.predict("This room is awesome !"))
