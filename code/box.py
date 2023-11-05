import torch
import torch.nn as nn


class Box:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub

    def check_postcondition(self, y) -> bool:
        target = y
        target_lb = self.lb[target].item()
        for i in range(self.ub.shape[-1]):
            if i != target and self.ub[i] >= target_lb:
                return False
        return True

    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> "Box":
        x = x.flatten()
        lb = x - eps
        lb.clamp_(min=0, max=1)

        ub = x + eps
        ub.clamp_(min=0, max=1)

        return Box(lb, ub)
