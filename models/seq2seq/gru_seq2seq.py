import numpy as np

import torch
import torch.nn as nn

from models.encoders.gru_encoder import GRUEncoder
from models.decoders.gru_decoder import GRUDecoder


class GRUSeq2Seq(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            output_dim,
            dec_emb_dim,
            dec_hid_dim,
            enc_pad_ix,
            enc_dropout=0.5,
            dec_dropout=0.5,
            device="cpu"
    ):
        super().__init__()

        self.encoder = GRUEncoder(
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            enc_pad_ix,
            dropout=enc_dropout
        )
        self.decoder = GRUDecoder(
            output_dim,
            dec_emb_dim,
            dec_hid_dim,
            dropout=dec_dropout
        )
        self.device = device

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        enc_outputs, context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input_ = trg[0]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input_, hidden, context)

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
