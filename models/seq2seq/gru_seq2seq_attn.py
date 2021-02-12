import numpy as np

import torch
import torch.nn as nn

from models.encoders.gru_bi_tokens_encoder import GRUBiTokensEncoder
from models.decoders.gru_attn_decoder import GRUAttnDecoder
from models.layers.attention import Attention


class GRUSeq2SeqAttn(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            enc_dropout,
            output_dim,
            dec_emb_dim,
            dec_hid_dim,
            dec_dropout,
            device
    ):
        super().__init__()

        self.encoder = GRUBiTokensEncoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.decoder = GRUAttnDecoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, self.attention, dec_dropout)
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input_ = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input_, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = np.random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_ = trg[t] if teacher_force else top1

        return outputs

    def predict(self, src, init_idx: int, max_trg_len: int):
        src = src.to(self.device)
        src_len, batch_size = src.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        enc_outputs, context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input_ = torch.zeros(batch_size, dtype=torch.long).to(self.device) + init_idx
        predictions = [input_]

        for t in range(1, max_trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input_, hidden, enc_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            top1 = output.argmax(1)

            input_ = top1
            predictions.append(input_)

        return torch.stack(predictions).to(self.device)
