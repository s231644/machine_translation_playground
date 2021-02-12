"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import torch
import torch.nn as nn
import dgl

from collections import namedtuple

from models.layers.treelstm_cells import TreeLSTMCell, ChildSumTreeLSTMCell

TreeLSTMBatch = namedtuple('SSTBatch', ['graph', 'mask', 'wordid'])


class TreeLSTMEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            emb_dim,
            hid_dim,
            enc_pad_ix,
            n_layers=1,
            dropout=0.5,
            cell_type='childsum',
            device="cpu"
    ):
        super(TreeLSTMEncoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.enc_pad_ix = enc_pad_ix

        self.device = device

        self.embedding = nn.Embedding(input_dim, emb_dim)
        cell = TreeLSTMCell if cell_type == 'nary' else ChildSumTreeLSTMCell
        self.rnn = cell(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: TreeLSTMBatch, g: dgl.DGLGraph, h, c):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : TreeLSTMBatch
            The data batch.
        g : dgl.DGLGraph
            Tree for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        h : Tensor
            Hidden states of each node.
        h0: root hidden states
        """
        # feed embedding
        embeds = self.dropout(self.embedding(batch.wordid * batch.mask))

        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = g.ndata.pop('h')
        return h, h[0]
