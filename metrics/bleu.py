# Source: https://github.com/fmcurti/pytorch-lightning/

from typing import List

import torch
from torchtext.data.metrics import bleu_score

from pytorch_lightning.metrics.metric import Metric


class BLEU(Metric):
    """
    Computes the Bleu score.
    Example:
    >>> candidate_corpus = [['My', 'full', 'pl', 'test'], ['Another', 'Sentence']]
    >>> references_corpus = [[['My', 'full', 'pl', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
    >>> metric = BLEU()
    >>> metric(candidate_corpus, references_corpus)
    tensor(0.8409)

    """

    def __init__(
            self,
            ignore_index: int,
            max_n: int = 4,
            weights: tuple = (0.25, 0.25, 0.25, 0.25),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_n = max_n
        self.weights = weights
        self.pad_idx = torch.tensor(ignore_index, requires_grad=False)

        self.add_state("targets", default=[])
        self.add_state("preds", default=[])

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        # trg_len, batch_size = target.shape

        preds = preds.T.cpu()
        target = target.T.cpu()

        for p, t in zip(preds, target):
            p_no_pad = p[p != self.pad_idx]
            t_no_pad = t[t != self.pad_idx]

            self.preds.append(self.tensor_to_str(p_no_pad))
            self.targets.append([self.tensor_to_str(t_no_pad)])

    def compute(self):
        return torch.tensor(
            bleu_score(self.preds, self.targets, self.max_n, self.weights)
        )

    @staticmethod
    def tensor_to_str(x: torch.Tensor) -> List[str]:
        assert len(x.shape) == 1, "Can only process 1d tensors"
        return list(map(str, x.detach().cpu().numpy().data))
