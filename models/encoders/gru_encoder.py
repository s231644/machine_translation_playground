import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            emb_dim,
            hid_dim,
            enc_pad_ix,
            n_layers=1,
            dropout=0.5,
            device="cpu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.enc_pad_ix = enc_pad_ix

        self.device = device

        self.embedding = nn.Embedding(input_dim, emb_dim).to(self.device)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

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
