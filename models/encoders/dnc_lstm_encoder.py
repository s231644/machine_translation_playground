import torch
import torch.nn as nn

from models.layers.dnc_lstm import DNCLSTM


class DNCLSTMEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            emb_dim,
            hid_dim,
            output_dim,
            enc_pad_ix,
            n_layers=1,
            n_heads=4,  # R
            n_words=25,  # N
            word_size=6,  # W
            dropout=0.5,
            device="cpu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.enc_pad_ix = enc_pad_ix

        self.device = device

        # input embedding
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = DNCLSTM(
            emb_dim, hid_dim,
            n_layers=n_layers,
            n_heads=n_heads, n_words=n_words, word_size=word_size,
            device=device
        )

        # # controller output to final output
        # self.Whry = nn.Linear(hid_dim + n_heads * word_size, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = src.to(self.device)
        src_len, batch_size = src.shape

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        # outputs - src_len x batch_size x hid_dim * n_directions

        # note: hidden is not _actually_ last state because of padding, let's find the real last state
        lengths = torch.as_tensor(
            src != self.enc_pad_ix, dtype=torch.int64
        ).sum(dim=0).clamp_max(src_len - 1)

        # lengths - batch_size

        last_state = outputs[lengths, torch.arange(batch_size)]
        # last_state = [batch size, hid dim * n directions]

        return outputs, last_state
