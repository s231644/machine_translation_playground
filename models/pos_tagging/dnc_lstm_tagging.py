import torch
import torch.nn as nn

from models.encoders.dnc_lstm_encoder import DNCLSTMEncoder


class DNCLSTMTagger(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            output_dim,
            enc_pad_ix,
            n_heads=4,
            n_words=25,
            word_size=6,
            enc_dropout=0.5,
            device="cpu"
    ):
        super().__init__()

        self.device = device

        self.encoder = DNCLSTMEncoder(
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            output_dim,
            enc_pad_ix,
            # n_layers=1,
            n_heads=n_heads,
            n_words=n_words,
            word_size=word_size,
            dropout=enc_dropout,
            device=device
        ).to(self.device)

        self.decoder = nn.Linear(enc_hid_dim + n_heads * word_size, output_dim).to(self.device)

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
