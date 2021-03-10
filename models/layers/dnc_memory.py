import torch
import torch.nn as nn


def oneplus(x):
    return 1 + torch.log(1 + torch.exp(x))


def outer_product(x, y):
    return torch.bmm(x.unsqueeze(-1), y.unsqueeze(-2))


def outer_sum(x, y):
    return x.unsqueeze(-1) + y.unsqueeze(-2)


def attention(memory, keys, betas):
    memory /= (memory.norm(dim=-1).unsqueeze(-1) + 1e-6)
    keys /= keys.norm(dim=1).unsqueeze(1)
    scores = memory @ keys  # batch_size x n_words x n_keys
    scores *= betas
    scores = torch.softmax(scores, dim=1)
    return scores


class DNCLSTMNet(nn.Module):
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

        self.n_heads = n_heads
        self.n_words = n_words
        self.word_size = word_size

        self.enc_pad_ix = enc_pad_ix

        self.device = device

        # input embedding
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # controller cell
        self.rnn_cell = nn.LSTMCell(emb_dim + n_heads * word_size, hid_dim)

        # controller output to final output
        self.Why = nn.Linear(hid_dim, output_dim)
        self.Wry = nn.Linear(n_heads * word_size, output_dim, bias=False)


        self.M = torch.zeros((n_words, word_size))  # memory
        self.u = torch.zeros(n_words)  # usage
        self.L = torch.zeros((n_words, n_words))  # linking
        self.p = torch.zeros(n_words)  # precedence_weight

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

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = src.to(self.device)
        src_len, batch_size = src.shape

        embedded = self.dropout(self.embedding(src))

        read_weights = torch.zeros((batch_size, self.n_words, self.n_heads))  # R x N
        write_weights = torch.zeros((batch_size, self.n_words))  # N

        read_vectors = torch.zeros((batch_size, self.word_size, self.n_heads))  # R x W

        u_t = torch.zeros((batch_size, self.n_words)) + 1e-6  # usage
        p_t = torch.zeros((batch_size, self.n_words))  # N
        memory_t = self.M.repeat(batch_size, 1, 1)
        links_t = torch.zeros((batch_size, self.n_words, self.n_words))  # R x R
        diag_mask = torch.eye(self.n_words).bool().unsqueeze(0).repeat(batch_size, 1, 1)

        h_t, s_t = None, None
        outputs_h, outputs_s, logits = [], [], []
        for t in range(src_len):
            # controller
            x_t = torch.cat([embedded[t], read_vectors.reshape(batch_size, -1)], dim=1)
            if t:
                h_t, s_t = self.rnn_cell(x_t, (h_t, s_t))
            else:
                h_t, s_t = self.rnn_cell(x_t)

            # xi
            free_gates, read_keys, read_strengths, read_modes = [], [], [], []
            for i in range(self.n_heads):
                free_gates.append(torch.sigmoid(self.Wh_free_gates[i](h_t)))  # 6
                read_keys.append(self.Wh_read_keys[i](h_t))  # 0
                read_strengths.append(oneplus(self.Wh_read_strengths[i](h_t)))  # 1
                read_modes.append(torch.softmax(self.Wh_read_modes[i](h_t), dim=-1))  # 9
            free_gates = torch.stack(free_gates, dim=-1)
            read_keys = torch.stack(read_keys, dim=-1)
            read_strengths = torch.stack(read_strengths, dim=-1)
            read_modes = torch.stack(read_modes, dim=-1)

            write_key = self.Wh_write_key(h_t)  # 2
            write_strength = oneplus(self.Wh_write_strength(h_t))  # 3

            erase_vector = torch.sigmoid(self.Wh_erase_vector(h_t))  # 4
            write_vector = torch.sigmoid(self.Wh_write_vector(h_t))  # 5

            allocation_gate = torch.sigmoid(self.Wh_allocation_gate(h_t))  # 7
            write_gate = torch.sigmoid(self.Wh_write_gate(h_t))  # 8

            psi_t = torch.exp(torch.sum(torch.log(1 - free_gates * read_weights), dim=-1))
            u_t = (u_t + write_weights - (u_t * write_weights)) * psi_t

            sorted_u_t, phi_t = torch.sort(u_t, dim=-1, descending=False)
            a_t = (1 - sorted_u_t) * torch.exp(torch.cumsum(torch.log(sorted_u_t), dim=-1) - torch.log(sorted_u_t))

            c_write_t = attention(memory_t, torch.unsqueeze(write_key, -1), torch.unsqueeze(write_strength, -1)).squeeze(-1)
            write_weights = write_gate * (allocation_gate * a_t + (1 - allocation_gate) * c_write_t)

            memory_t += outer_product(write_weights, write_vector - erase_vector)

            p_t = (1 - write_weights.sum(dim=1, keepdim=True)) * p_t + write_weights

            links_t *= 1 - outer_sum(write_weights, write_weights)
            links_t += outer_product(write_weights, p_t)
            links_t[diag_mask] = 0

            c_write_t = attention(memory_t, read_keys, read_strengths)
            f_t = links_t @ read_weights
            b_t = links_t.permute(0, 2, 1) @ read_weights

            read_weights = (read_modes.unsqueeze(dim=2) * torch.stack([b_t, c_write_t, f_t], dim=1)).sum(dim=1)
            read_vectors = memory_t.permute(0, 2, 1) @ read_weights
            # output
            y_t = self.Why(h_t) + self.Wry(read_vectors.reshape(batch_size, -1))

            outputs_h.append(h_t)
            outputs_s.append(s_t)
            logits.append(y_t)

        return torch.stack(logits)


model = DNCLSTMNet(5, 4, 3, 7, 0)
x = torch.LongTensor(
    [
     [3, 3, 1, 2, 0],
     [1, 2, 3, 4, 1]
    ]
).T
print(model(x))
