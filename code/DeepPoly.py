from collections import namedtuple
import torch
import torch.nn as nn
from typing import List, Tuple

from box import AbstractBox


LinearBound = namedtuple("LinearBound", ["lower", "upper"])


class DeepPoly:
    def __init__(self, x: torch.Tensor, eps: float):
        self.initial_box = AbstractBox.construct_initial_box(x, eps)
        # a list of Boxes; each element stores the concrete bounds for one layer
        self.boxes: List[AbstractBox] = []
        # a list of LinearBounds; each element stores the linear bounds for one layer
        self.linear_bounds: List[LinearBound] = []
        # If the bounds are stored as above, then every propagate method
        # should append a box and a linear bound to the lists above.


    def check_postcondition(self, y) -> bool:
        target = y
        target_lb = self.lb[0][target].item()
        for i in range(self.ub.shape[-1]):
            if i != target and self.ub[0][i] >= target_lb:
                return False
        return True

    def backsubstitute(self, layer_number: int) -> AbstractBox:
        """
        Performs backsubstitution to compute bounds for a given layer.

        Args:
            layer_number: index of the layer for which to compute bounds
        """
        # compute linear bound wrt network input
        linb = self.linear_bounds[layer_number]
        for previous_linb in reversed(self.linear_bounds[:-layer_number]):
            linb.lower = (
                torch.maximum(torch.tensor(0), linb.lower) @ previous_linb.lower
                + torch.minimum(torch.tensor(0), linb.lower) @ previous_linb.upper
            )
            linb.upper = (
                torch.maximum(torch.tensor(0), linb.upper) @ previous_linb.upper
                + torch.minimum(torch.tensor(0), linb.upper) @ previous_linb.lower
            )

        initial_lb, initial_ub = self.initial_box.lb, self.initial_box.ub
        initial_lb = torch.cat([initial_lb, torch.tensor([1])])  # append 1 for bias
        initial_ub = torch.cat([initial_ub, torch.tensor([1])])  # append 1 for bias
        lb = (
            torch.maximum(torch.tensor(0), linb.lower) @ initial_lb
            + torch.minimum(torch.tensor(0), linb.upper) @ initial_ub
        )
        ub = (
            torch.maximum(torch.tensor(0), linb.upper) @ initial_ub
            + torch.minimum(torch.tensor(0), linb.lower) @ initial_lb
        )

        return AbstractBox(lb, ub)

    def propagate_linear(self, linear: nn.Linear):
        linear_bound = torch.cat([linear.weight, linear.bias])
        # Lower and upper linear bounds are identical for linear layer
        linear_bound = LinearBound(linear_bound, linear_bound)
        self.linear_bounds.append(linear_bound)

        box = self.backsubstitute(-1)
        self.boxes.append(box)

    def propagate_flatten(self, flatten: nn.Flatten):
        linear_bound = 
        box = self.backsubstitute(-1)
        self.boxes.append(box)

    # TODO: Write methods to propagate through different modules


def certify_sample(model, x, y, eps) -> bool:
    box = DeepPoly.propagate_sample(model, x, eps)
    return box.check_postcondition(y)

def propagate_sample(model, x, eps) -> AbstractBox:
    pass  # TODO: Implement
    dp = DeepPoly(x, eps)
    for layer in model:
        if isinstance(layer, nn.Linear):
            dp.propagate_linear(layer)
        elif isinstance(layer, nn.ReLU):
            dp.propagate_relu(layer)
        elif isinstance(layer, nn.Flatten):
            dp.propagate_flatten(layer)
        elif isinstance(layer, nn.LeakyReLU):
            dp.propagate_leaky_relu(layer)
        elif isinstance(layer, nn.Conv2d):
            dp.propagate_conv2d(layer)
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
    return dp.boxes[-1]