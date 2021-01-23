import torch
import torch.nn as nn


class GRUDecoder(nn.Module):
    def __init__(
            self,
            output_dim,
            emb_dim,
            hid_dim,
            dropout=0.5,
            device="cpu"
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.device = device

        self.embedding = nn.Embedding(output_dim, emb_dim).to(self.device)
        self.rnn_cell = nn.GRUCell(emb_dim + hid_dim, hid_dim).to(self.device)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

    def forward(self, input_, hidden, context):
        embedded = self.dropout(self.embedding(input_.to(self.device)))
        emb_con = torch.cat((embedded, context.to(self.device)), dim=1)
        hidden = self.rnn_cell(emb_con, hidden.to(self.device))
        output = torch.cat((embedded, hidden, context), dim=1)
        prediction = self.fc_out(output)

        return prediction, hidden
