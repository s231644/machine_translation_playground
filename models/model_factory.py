from models.model_seq2seq_lstm import Seq2SeqLSTM
from models.model_seq2seq_gru import Seq2SeqGRU
from models.model_seq2seq_gru_attention import Seq2SeqAtt
from models.model_seq2seq_gru_attention_pad_mask import Seq2SeqAttPadMask
from models.model_seq2seq_cnn import Seq2SeqCNN

from utils.utils import init_weights_uniform, init_weights_normal


def get_model(model_name, **kwargs):
    if model_name == 'seq2seq_lstm':
        model = Seq2SeqLSTM(**kwargs)
    elif model_name == 'seq2seq_gru':
        model = Seq2SeqGRU(**kwargs)
    elif model_name == 'seq2seq_gru_attention':
        model = Seq2SeqAtt(**kwargs)
    elif model_name == 'seq2seq_gru_attention_pad_mask':
        model = Seq2SeqAttPadMask(**kwargs)
    elif model_name == 'seq2seq_cnn':
        model = Seq2SeqCNN(**kwargs)
    else:
        raise ValueError

    return model


def model_init_weights(model_name, model):
    if model_name == 'seq2seq_lstm':
        model = model.apply(init_weights_uniform)
    elif model_name == 'seq2seq_gru':
        model = model.apply(init_weights_normal)
    elif model_name == 'seq2seq_gru_attention':
        model = model.apply(init_weights_normal)
    elif model_name == 'seq2seq_gru_attention_pad_mask':
        model = model.apply(init_weights_normal)
    elif model_name == 'seq2seq_cnn':
        model = model
    else:
        raise ValueError

    return model

