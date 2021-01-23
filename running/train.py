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

from models.model_factory import get_model, model_init_weights
from utils.utils import count_parameters
from utils.dataset import data_train_test_split, preprocessing
from runner.runner import Runner


# set seed
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# split data
SRC, TRG, train_data, val_data, test_data = data_train_test_split(with_pad=params['with_pad'],
                                                                  inversion=params['inversion_dataset'],
                                                                  batch_first=params['batch_first'])
# create iterators
train_iterator, val_iterator, test_iterator = preprocessing(batch_size=params['train_params']['batch_size'],
                                                            device=device,
                                                            with_pad=params['with_pad'])

# prepare model
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

model = get_model(**model_params)
model = model.to(device)
model = model_init_weights(params['model_name'], model)

print(f'The model has {count_parameters(model):,} trainable parameters')

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

runner = Runner(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    train_iterator=train_iterator,
    val_iterator=val_iterator,
    trg_pad_idx=TRG_PAD_IDX,
    epoch_size=0,
    batch_size=64,
)

checkpoint_callback = ModelCheckpoint(
    filepath=str(Path(params['logs_dir']).joinpath(params['model_name']).joinpath('weights')),
    verbose=params['checkpoint_callback']['verbose'],
    monitor=params['checkpoint_callback']['monitor'],
    mode=params['checkpoint_callback']['mode'],
    save_top_k=params['checkpoint_callback']['save_top_k'],
)

logger = TensorBoardLogger(
    Path(params['logs_dir']),
    name=params['model_name']
)

trainer = Trainer(
    checkpoint_callback=checkpoint_callback,
    logger=logger,
    gpus=[params['ngpu']] if params['cuda'] else 0,
    log_gpu_memory='all',
    max_epochs=params['train_params']['n_epochs'],
    num_sanity_val_steps=params['train_params']['batch_size']
)

trainer.fit(runner)
