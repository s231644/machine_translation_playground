import torch
import torch.nn as nn


class GRUDecoder(nn.Module):
    def __init__(
            self,
            output_dim,
            emb_dim,
            hid_dim,
            dropout=0.5
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn_cell = nn.GRUCell(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_, hidden, context):
        embedded = self.dropout(self.embedding(input_))
        emb_con = torch.cat((embedded, context), dim=1)
        hidden = self.rnn_cell(emb_con, hidden)
        output = torch.cat((embedded, hidden, context), dim=1)
        prediction = self.fc_out(output)

        return prediction, hidden
