import torch
from pytorch_lightning.metrics import Metric


class ExactMatch(Metric):
    def __init__(self, ignore_index, **kwargs):
        super().__init__(**kwargs)
        self.pad_idx = torch.tensor(ignore_index, requires_grad=False)
        self.zero = torch.tensor(0, requires_grad=False)
        self.one = torch.tensor(1, requires_grad=False)

        self.add_state("correct", default=self.zero, dist_reduce_fx="sum")
        self.add_state("total", default=self.zero, dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        loss_mask = torch.where(
            torch.BoolTensor(target.cpu() == self.pad_idx), self.zero, self.one
        )
        equal_mask = torch.where(
            torch.BoolTensor(target.cpu() != preds.cpu()), self.one, self.zero
        )

        corrects = torch.sum(loss_mask * equal_mask, dim=0)

        self.correct += torch.sum(
            torch.BoolTensor(corrects == self.zero)
        )
        self.total += len(corrects)

    def compute(self):
        return self.correct.float() / self.total
