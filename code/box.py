import torch
import torch.nn as nn

class AbstractBox:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub


    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'AbstractBox':
        lb = x - eps
        lb.clamp_(min=0, max=1)

        ub = x + eps
        ub.clamp_(min=0, max=1)

        return AbstractBox(lb, ub)
