import torch
import torch.nn as nn


class FactorizedGRUBiTokensEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            emb_dim,
            enc_hid_dim,
            dec_hid_dim,
            dropout=0.5
    ):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.ohe_embedding = nn.Embedding(input_dim, input_dim, _weight=torch.eye(input_dim))
        for param in self.ohe_embedding.parameters():
            param.requires_grad = False

        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        # semantic information
        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        # syntactic information
        ohe_embedded = self.ohe_embedding(src)
        outputs, hidden = self.rnn(ohe_embedded)

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return embedded, outputs, hidden

