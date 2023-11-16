import torch
import torch.nn as nn


class Box:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub

    def check_postcondition(self) -> bool:
        return (self.lb >= 0).all()

    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> "Box":
        x = x.flatten()
        lb = (x - eps).clamp_(min=0, max=1)
        ub = (x + eps).clamp_(min=0, max=1)

        return Box(lb, ub)
