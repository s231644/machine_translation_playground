import torch
import torch.nn as nn


class FactorizedGRUAttnDecoder(nn.Module):
    def __init__(
            self,
            output_dim,
            emb_dim,
            enc_hid_dim,
            dec_hid_dim,
            attention,
            dropout=0.5,
            device="cpu"
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim

        self.device = device

        self.embedding = nn.Embedding(output_dim, emb_dim).to(self.device)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim).to(self.device)
        self.attention = attention.to(self.device)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

    def forward(self, input_, hidden, encoder_embeddings, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # semantic_embeddings = [src len, batch size, enc hid dim * 2]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input_ = input_.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input_))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_embeddings = encoder_embeddings.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_embeddings)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)
