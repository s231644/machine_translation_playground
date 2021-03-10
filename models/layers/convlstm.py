# based on https://openreview.net/pdf?id=BylWglrYPH
# https://github.com/mitchelljeff/symmetry/blob/master/palindrome/convlstm.py

import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    # cf. appendix B
    def __init__(self, emb_dim, hid_dim, stack_size=-1, device="cpu"):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.stack_size = stack_size

        self.device = device

        self.wf = nn.Linear(emb_dim + hid_dim, hid_dim * 3, bias=True).to(device)
        self.wi = nn.Linear(emb_dim + hid_dim, hid_dim, bias=True).to(device)
        self.wc = nn.Linear(emb_dim + hid_dim, hid_dim, bias=True).to(device)
        self.wo = nn.Linear(emb_dim + hid_dim, hid_dim, bias=True).to(device)

        up = torch.stack(
            (torch.eye(hid_dim), torch.zeros((hid_dim, hid_dim)), torch.zeros((hid_dim, hid_dim))), dim=-1
        )
        stay = torch.stack(
            (torch.zeros((hid_dim, hid_dim)), torch.eye(hid_dim), torch.zeros((hid_dim, hid_dim))), dim=-1
        )
        down = torch.stack(
            (torch.zeros((hid_dim, hid_dim)), torch.zeros((hid_dim, hid_dim)), torch.eye(hid_dim)), dim=-1
        )

        self.conv_up = nn.Conv1d(
            hid_dim, hid_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False
        )
        self.conv_up.weight = nn.Parameter(up, requires_grad=True)
        # self.conv_up.bias = 0
        self.conv_up.to(device)

        self.conv_stay = nn.Conv1d(
            hid_dim, hid_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
        )
        self.conv_stay.weight = nn.Parameter(stay, requires_grad=True)
        # self.conv_stay.bias = 0
        self.conv_stay.to(device)

        self.conv_down = nn.Conv1d(
            hid_dim, hid_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
        )
        self.conv_down.weight = nn.Parameter(down, requires_grad=True)
        # self.conv_down.bias = 0
        self.conv_down.to(device)

    def forward(self, x):
        # [x] --- seq_len x batch_size x emb_dim
        seq_len, batch_size = x.shape[:2]
        h = torch.zeros((batch_size, self.hid_dim)).to(self.device)
        state = torch.zeros((batch_size, self.hid_dim, 0)).to(self.device)

        outputs = torch.zeros((seq_len, batch_size, self.hid_dim)).to(self.device)

        for t in range(seq_len):
            g_t = torch.cat([h, x[t]], dim=-1)  # batch_size x (hid_dim + emb_dim)

            # forget gate
            f_t = self.wf(g_t)  # batch_size x (hid_dim * 3)
            f_t = torch.reshape(f_t, (-1, self.hid_dim, 1, 3))  # batch_size x hid_dim x 1 x 3
            f_t = torch.softmax(f_t, dim=-1)

            # input gate
            i_t = torch.sigmoid(self.wi(g_t))  # batch_size x hid_dim

            # output gate
            o_t = torch.sigmoid(self.wo(g_t))  # batch_size x hid_dim

            # control gate
            c_hat_t = torch.tanh(self.wc(g_t))  # batch_size x hid_dim

            i_upd_t = i_t * c_hat_t

            # new ideas
            state = torch.cat([state, torch.unsqueeze(i_upd_t, 2)], dim=-1)

            state_up = self.conv_up(state)
            state_stay = self.conv_stay(state)
            state_down = self.conv_down(state)

            state = torch.stack([state_up, state_stay, state_down], dim=-1)
            state = torch.sum(state * f_t, dim=-1)

            h = state[:, :, -1] * o_t
            outputs[t] = h
        return outputs, h
