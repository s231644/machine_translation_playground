import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return torch.softmax(attention, dim=1)


class AttentionLayerY(nn.Module):
    def __init__(self, name, enc_size, dec_size, hid_size, activ=torch.tanh):
        """ A layer that computes additive attention response and weights """
        super().__init__()
        self.name = name
        self.enc_size = enc_size  # num units in encoder state
        self.dec_size = dec_size  # num units in decoder state
        self.hid_size = hid_size  # attention layer hidden units
        self.activ = activ  # attention layer hidden nonlinearity

        # create trainable paramteres like this:
        self.linear_enc = nn.Linear(self.enc_size, self.hid_size)
        self.linear_dec = nn.Linear(self.dec_size, self.hid_size)
        # self.linear_out = nn.Linear(2 * self.hid_size, 1)
        self.linear_out = nn.Linear(self.hid_size, 1)

    def forward(self, enc, dec, inp_mask=None):
        """
        Computes attention response and weights
        :param enc: encoder activation sequence, float32[batch_size, ninp, enc_size]
        :param dec: single decoder state used as "query", float32[batch_size, dec_size]
        :param inp_mask: mask on enc activatons (0 after first eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - attention response vector (weighted sum of enc)
            - probs - attention weights after softmax
        """

        # Compute logits
        hidden_enc = self.linear_enc(enc)  # [ninp, hid_size]
        hidden_dec = self.linear_dec(dec)  # [1, dec_size]

        # hidden_enc_dec = torch.cat([hidden_enc, hidden_dec.repeat(enc.shape[0], 1)], dim=1)
        hidden_enc_dec = hidden_enc + hidden_dec.repeat(enc.shape[0], 1)

        logits = self.linear_out(self.activ(hidden_enc_dec))  # [ninp, 1]
        logits = torch.squeeze(logits, -1)  # [ninp]

        # Apply mask - if mask is 0, logits should be -inf or -1e9
        # You may need torch.where
        if inp_mask is not None:
            logits = torch.where(inp_mask > 0, logits, torch.tensor(-1e9))

        # Compute attention probabilities (softmax)
        probs = torch.softmax(logits)

        # Compute attention response using enc and probs
        attn = torch.unsqueeze(torch.sum(torch.unsqueeze(probs, 1) * hidden_enc, 0, keepdims=True), 0)

        return attn, probs
