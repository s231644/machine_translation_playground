import torch
import numpy as np
import random
import argparse
import yaml

from models.model_seq2seq_lstm import Seq2SeqLSTM
from models.model_seq2seq_gru_attention_pad_mask import Seq2SeqAttPadMask
from utils.translation import translate_sentence, translate_sentence_with_attention
from utils.dataset import data_train_test_split, preprocessing


# set seed
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# parser
parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default='configs/seq2seq_lstm.yaml', help='')
parser.add_argument('--config', type=str, default='configs/seq2seq_gru_attention_pad_mask.yaml', help='')
params = parser.parse_args()

try:
    params = yaml.full_load(open(params.config))
except FileNotFoundError:
    print("No such file or directory: {}".format(params.config))
    exit(1)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# split data
SRC, TRG, train_data, val_data, test_data = data_train_test_split(params['base_model']['with_pad'])
# create iterators
train_iterator, val_iterator, test_iterator = preprocessing(batch_size=params['train_params']['batch_size'],
                                                            device=device,
                                                            with_pad=params['base_model']['with_pad'])

# prepare model
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

# model = Seq2SeqLSTM(input_dim=INPUT_DIM,
#                     enc_emb_dim=params['base_model']['enc_emb_dim'],
#                     enc_dropout=params['base_model']['enc_dropout'],
#                     output_dim=OUTPUT_DIM,
#                     dec_emb_dim=params['base_model']['dec_emb_dim'],
#                     dec_dropout=params['base_model']['dec_dropout'],
#                     hid_dim=params['base_model']['hid_dim'],
#                     n_layers=params['base_model']['n_layers'],
#                     device=device)

model = Seq2SeqAttPadMask(input_dim=INPUT_DIM,
                          emb_dim=params['base_model']['emb_dim'],
                          enc_hid_dim=params['base_model']['enc_hid_dim'],
                          dec_hid_dim=params['base_model']['dec_hid_dim'],
                          output_dim=OUTPUT_DIM,
                          dropout=params['base_model']['dropout'],
                          src_pad_idx=SRC_PAD_IDX,
                          device=device)

# pretrained_dict = torch.load('logs/seq2seq_lstm/weights-v1.ckpt')
pretrained_dict = torch.load('logs/seq2seq_attention_pad_mask/weights-v1.ckpt')

if 'state_dict' in pretrained_dict.keys():
    pretrained_dict = pretrained_dict['state_dict']

model_dict = model.state_dict()

new_state = {}
for k, v in model_dict.items():
    if 'model.' + k in pretrained_dict.keys() and pretrained_dict['model.' + k].size() == v.size():
        # print('model.' + k)
        new_state[k] = pretrained_dict['model.' + k]
    else:
        new_state[k] = model_dict[k]
model.load_state_dict(new_state)
model = model.to(device)

example_idx = 12

src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')

# translation = translate_sentence(src, SRC, TRG, model, device)
translation, attention = translate_sentence_with_attention(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')
