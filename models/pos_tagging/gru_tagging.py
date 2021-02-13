import numpy as np

import torch
import torch.nn as nn

from models.encoders.gru_encoder import GRUEncoder


class GRUTagger(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            output_dim,
            enc_pad_ix,
            enc_dropout=0.5,
            device="cpu"
    ):
        super().__init__()

        self.device = device

        self.encoder = GRUEncoder(
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            enc_pad_ix,
            dropout=enc_dropout,
            device=device
        ).to(self.device)

        self.decoder = nn.Linear(enc_hid_dim, output_dim).to(self.device)

    def forward(self, src):
        src = src.to(self.device)

        # last hidden state of the encoder is the context
        enc_outputs, context = self.encoder(src)
        outputs = self.decoder(enc_outputs)

        return outputs

    def predict(self, src):
        src = src.to(self.device)

        # last hidden state of the encoder is the context
        enc_outputs, context = self.encoder(src)

        outputs = self.decoder(enc_outputs)
        predictions = torch.argmax(outputs, dim=-1)

        return predictions
