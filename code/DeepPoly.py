from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.autograd.functional import jacobian
from torch.nn.functional import relu
import torch.nn as nn

from Box import Box
from utils.attach_shapes import attach_shapes


@dataclass(frozen=True)
class LinearBound:
    lower_weight: torch.Tensor
    upper_weight: torch.Tensor
    lower_bias: Optional[torch.Tensor] = None
    upper_bias: Optional[torch.Tensor] = None

    def get_params(self):
        return self.lower_weight.clone(), self.upper_weight.clone(), self.lower_bias.clone(), self.upper_bias.clone()


class DeepPoly:
    def __init__(self, x: torch.Tensor, eps: float):
        self.initial_box = Box.construct_initial_box(x, eps)
        # a list of Boxes; each element stores the concrete bounds for one layer
        self.boxes: List[Box] = []
        # a list of LinearBounds; each element stores the linear bounds for one layer
        self.linear_bounds: List[LinearBound] = []
        # If the bounds are stored as above, then every propagate method
        # should append a box and a linear bound to the lists above.

    def backsubstitute(self, layer_number: int) -> Box:
        """
        Performs backsubstitution to compute bounds for a given layer.

        Args:
            layer_number: index of the layer for which to compute bounds
        """
        # compute linear bound wrt network input
        linb = self.linear_bounds[layer_number]
        lower_weight, upper_weight, lower_bias, upper_bias = linb.get_params()

        for prev_linb in reversed(self.linear_bounds[:layer_number]):
            if prev_linb.lower_bias is not None:
                if lower_bias is None:
                    lower_bias = upper_bias = torch.zeros(lower_weight.shape[0])
                lower_bias += relu(lower_weight) @ prev_linb.lower_bias - relu(-lower_weight) @ prev_linb.upper_bias
                upper_bias += relu(upper_weight) @ prev_linb.upper_bias - relu(-upper_weight) @ prev_linb.lower_bias

            lower_weight = relu(lower_weight) @ prev_linb.lower_weight - relu(-lower_weight) @ prev_linb.upper_weight
            upper_weight = relu(upper_weight) @ prev_linb.upper_weight - relu(-upper_weight) @ prev_linb.lower_weight

        # Insert the initial boxes into the linear bounds
        initial_lb, initial_ub = self.initial_box.lb, self.initial_box.ub
        lb = relu(lower_weight) @ initial_lb - relu(-lower_weight) @ initial_ub
        ub = relu(upper_weight) @ initial_ub - relu(-upper_weight) @ initial_lb

        lb = lb + lower_bias
        ub = ub + upper_bias

        return Box(lb, ub)

    def propagate_linear(self, linear: nn.Linear):
        linear_bound = linear.weight
        bias = linear.bias
        # Lower and upper linear bounds are identical for linear layer
        linear_bound = LinearBound(linear_bound, linear_bound, bias, bias)
        self.linear_bounds.append(linear_bound)

        box = self.backsubstitute(-1)
        self.boxes.append(box)

    def propagate_conv(self, conv: nn.Conv2d):
        x = torch.rand(conv.input_shape)
        W = jacobian(conv, x)
        W = W.reshape(conv.out_features, conv.in_features)
        if conv.bias is not None:
            # Conv2d saves bias as a tensor of shape (out_channels,)
            b = conv.bias.repeat_interleave(int(conv.out_features / conv.out_channels))
        else:
            b = None
        linear_bound = LinearBound(W, W, b, b)
        self.linear_bounds.append(linear_bound)

        box = self.backsubstitute(-1)
        self.boxes.append(box)

    def propagate_relu(self, relu: nn.ReLU):
        pass  # TODO

    def propagate_leaky_relu(self, leaky_relu: nn.LeakyReLU):
        pass  # TODO

    # TODO: Write methods to propagate through different modules


def certify_sample(model, x, y, eps) -> bool:
    for param in model.parameters():
        param.requires_grad = False
    attach_shapes(model, x.unsqueeze(0).shape)
    box = propagate_sample(model, x, eps)
    return box.check_postcondition(y)


def propagate_sample(model, x, eps) -> Box:
    # TODO: Implement
    dp = DeepPoly(x, eps)
    for layer in model:
        if isinstance(layer, nn.Linear):
            dp.propagate_linear(layer)
        elif isinstance(layer, nn.ReLU):
            dp.propagate_relu(layer)
        elif isinstance(layer, nn.Flatten):
            continue
        elif isinstance(layer, nn.LeakyReLU):
            dp.propagate_leaky_relu(layer)
        elif isinstance(layer, nn.Conv2d):
            dp.propagate_conv(layer)
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
    return dp.boxes[-1]
