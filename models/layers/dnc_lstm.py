import torch
import torch.nn as nn


class DNCLSTM(nn.Module):
    def __init__(
            self, emb_dim, hid_dim, n_layers=1,
            n_heads=4, n_words=25, word_size=6,
            device="cpu"
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.n_heads = n_heads
        self.n_words = n_words
        self.word_size = word_size

        self.device = device

        # controller cell
        self.rnn_cell = nn.LSTMCell(emb_dim + n_heads * word_size, hid_dim)

        # memory
        self.M = torch.zeros((n_words, word_size))

        # xi --- interface vector
        # 0
        self.Wh_read_keys = nn.ModuleList([nn.Linear(hid_dim, word_size) for i in range(n_heads)])
        # 1
        self.Wh_read_strengths = nn.ModuleList([nn.Linear(hid_dim, 1) for i in range(n_heads)])
        # 2
        self.Wh_write_key = nn.Linear(hid_dim, word_size)
        # 3
        self.Wh_write_strength = nn.Linear(hid_dim, 1)
        # 4
        self.Wh_erase_vector = nn.Linear(hid_dim, word_size)
        # 5
        self.Wh_write_vector = nn.Linear(hid_dim, word_size)
        # 6
        self.Wh_free_gates = nn.ModuleList([nn.Linear(hid_dim, 1) for i in range(n_heads)])
        # 7
        self.Wh_allocation_gate = nn.Linear(hid_dim, 1)
        # 8
        self.Wh_write_gate = nn.Linear(hid_dim, 1)
        # 9
        self.Wh_read_modes = nn.ModuleList([nn.Linear(hid_dim, 3) for i in range(n_heads)])

    def forward(self, x):
        # [x] --- seq_len x batch_size x emb_dim
        seq_len, batch_size = x.shape[:2]

        read_weights = torch.zeros((batch_size, self.n_words, self.n_heads))  # R x N
        write_weights = torch.zeros((batch_size, self.n_words))  # N
        read_vectors = torch.zeros((batch_size, self.word_size, self.n_heads))  # R x W
        r_t = read_vectors.reshape(batch_size, -1)
        u_t = torch.zeros((batch_size, self.n_words)) + 1e-6  # usage
        p_t = torch.zeros((batch_size, self.n_words))  # N
        memory_t = self.M.repeat(batch_size, 1, 1)
        links_t = torch.zeros((batch_size, self.n_words, self.n_words))  # R x R
        diag_mask = torch.eye(self.n_words).bool().unsqueeze(0).repeat(batch_size, 1, 1)
        # h_t, s_t, hr_t = None, None, None

        h_t = torch.zeros((batch_size, self.hid_dim))
        s_t = torch.zeros((batch_size, self.hid_dim))
        hr_t = torch.zeros((batch_size, self.hid_dim + self.word_size * self.n_heads))

        outputs_h, outputs_s, outputs_hr = [], [], []
        for t in range(seq_len):
            # controller
            x_t = torch.cat([x[t], r_t], dim=-1)
            if t:
                h_t, s_t = self.rnn_cell(x_t, (h_t, s_t))
            else:
                h_t, s_t = self.rnn_cell(x_t)

            # xi
            free_gates, read_keys, read_strengths, read_modes = [], [], [], []
            for i in range(self.n_heads):
                free_gates.append(torch.sigmoid(self.Wh_free_gates[i](h_t)))  # 6
                read_keys.append(self.Wh_read_keys[i](h_t))  # 0
                read_strengths.append(self.oneplus(self.Wh_read_strengths[i](h_t)))  # 1
                read_modes.append(torch.softmax(self.Wh_read_modes[i](h_t), dim=-1))  # 9
            free_gates = torch.stack(free_gates, dim=-1)
            read_keys = torch.stack(read_keys, dim=-1)
            read_strengths = torch.stack(read_strengths, dim=-1)
            read_modes = torch.stack(read_modes, dim=-1)

            write_key = self.Wh_write_key(h_t)  # 2
            write_strength = self.oneplus(self.Wh_write_strength(h_t))  # 3

            erase_vector = torch.sigmoid(self.Wh_erase_vector(h_t))  # 4
            write_vector = torch.sigmoid(self.Wh_write_vector(h_t))  # 5

            allocation_gate = torch.sigmoid(self.Wh_allocation_gate(h_t))  # 7
            write_gate = torch.sigmoid(self.Wh_write_gate(h_t))  # 8

            psi_t = torch.exp(torch.sum(torch.log(1 - free_gates * read_weights), dim=-1))
            u_t = (u_t + write_weights - (u_t * write_weights)) * psi_t

            sorted_u_t, phi_t = torch.sort(u_t, dim=-1, descending=False)
            a_t = (1 - sorted_u_t) * torch.exp(torch.cumsum(torch.log(sorted_u_t), dim=-1) - torch.log(sorted_u_t))

            c_write_t = self.attention(memory_t, torch.unsqueeze(write_key, -1), torch.unsqueeze(write_strength, -1)).squeeze(-1)
            write_weights = write_gate * (allocation_gate * a_t + (1 - allocation_gate) * c_write_t)

            memory_t = memory_t + self.outer_product(write_weights, write_vector - erase_vector)

            p_t = (1 - write_weights.sum(dim=1, keepdim=True)) * p_t + write_weights

            links_t = ((1 - self.outer_sum(write_weights, write_weights)) * links_t
                       + self.outer_product(write_weights, p_t))
            links_t = torch.where(diag_mask, torch.tensor(0.0), links_t)

            c_write_t = self.attention(memory_t, read_keys, read_strengths)
            f_t = links_t @ read_weights
            b_t = links_t.permute(0, 2, 1) @ read_weights

            read_weights = (read_modes.unsqueeze(dim=2) * torch.stack([b_t, c_write_t, f_t], dim=1)).sum(dim=1)
            read_vectors = memory_t.permute(0, 2, 1) @ read_weights

            r_t = read_vectors.reshape(batch_size, -1)
            hr_t = torch.cat([h_t, r_t], dim=-1)
            outputs_h.append(h_t)
            outputs_s.append(s_t)
            outputs_hr.append(hr_t)

        return torch.stack(outputs_hr), hr_t

    @staticmethod
    def oneplus(x):
        return 1 + torch.log(1 + torch.exp(x))

    @staticmethod
    def outer_product(x, y):
        return torch.bmm(x.unsqueeze(-1), y.unsqueeze(-2))

    @staticmethod
    def outer_sum(x, y):
        return x.unsqueeze(-1) + y.unsqueeze(-2)

    @staticmethod
    def attention(memory, keys, betas):
        memory = memory / (memory.norm(dim=-1).unsqueeze(-1) + 1e-6)
        keys = keys / keys.norm(dim=1).unsqueeze(1)
        scores = memory @ keys  # batch_size x n_words x n_keys
        scores *= betas
        scores = torch.softmax(scores, dim=1)
        return scores
