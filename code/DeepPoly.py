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
        ilb, iub = self.initial_box.lb, self.initial_box.ub
        lb = relu(lower_weight) @ ilb - relu(-lower_weight) @ iub
        ub = relu(upper_weight) @ iub - relu(-upper_weight) @ ilb
        if lower_bias is not None:
            lb += lower_bias
            ub += upper_bias

        return Box(lb, ub)

    def propagate_linear(self, linear: nn.Linear):
        W = linear.weight
        b = linear.bias
        linear_bound = LinearBound(W, W, b, b)
        self.linear_bounds.append(linear_bound)

        box = self.backsubstitute(-1)
        self.boxes.append(box)

    def propagate_conv(self, conv: nn.Conv2d):
        x = torch.rand(conv.input_shape)
        J = jacobian(conv, x)  # convenient to avoid building W manually, but loses exactness due to autodiff
        W = J.reshape(conv.out_features, conv.in_features)
        # Conv2d saves bias as a tensor of shape (out_channels,)
        assert conv.out_features % conv.out_channels == 0
        b = conv.bias.repeat_interleave(conv.out_features // conv.out_channels) if conv.bias is not None else None
        linear_bound = LinearBound(W, W, b, b)
        self.linear_bounds.append(linear_bound)

        box = self.backsubstitute(-1)
        self.boxes.append(box)

    def propagate_relu(self, relu: nn.ReLU):
        """
        Case 1: ub < 0 -> linear bounds = 0 (= lb = ub)
        Case 2: lb > 0 -> linear bounds = x_{i-1} (lb = lb_{i-1}, ub = ub_{i-1})
        Case 3: mixed
            - lambda (slope) = ub_{i-1} / (ub_{i-1} - lb_{i-1})
            - a) u <= -l -> Relaxation 1: linear_lb = 0, linear_ub = lambda * (x_{i-1}-lb_{i-1}) (lb = 0, ub = ub_{i-1})
            - b) u >  -l -> Relaxation 2: linear_lb = x_{i-1}, linear_ub = lambda * (x_{i-1}-lb_{i-1}) (lb = lb_{i-1}, ub = ub_{i-1})

        Args:
            relu:
        """
        # NOTE: could use leaky relu function also for relu. Note that the current leaky_relu propagator always
        # chooses relaxation 1.

        # case_1, case_2, case_3a, case_3b
        lower_bound = torch.stack(
            [
                torch.zeros(self.linear_bounds[-1].lower_weight.shape[0]),
                torch.ones(self.linear_bounds[-1].lower_weight.shape[0]),
                torch.zeros(self.linear_bounds[-1].lower_weight.shape[0]),
                torch.ones(self.linear_bounds[-1].lower_weight.shape[0]),
            ],
            dim=1,
        )

        bound = self.boxes[-1].ub / (self.boxes[-1].ub - self.boxes[-1].lb)
        upper_bound = torch.stack(
            [
                torch.zeros(self.linear_bounds[-1].upper_weight.shape[0]),
                torch.ones(self.linear_bounds[-1].upper_weight.shape[0]),
                bound.clone().detach(),
                bound.clone().detach(),
            ],
            dim=1,
        )
        bias = -bound * self.boxes[-1].lb
        lower_bias = torch.zeros(self.linear_bounds[-1].lower_bias.shape[0], 4)
        upper_bias = torch.cat(
            [
                torch.zeros(self.linear_bounds[-1].lower_bias.shape[0], 2),
                bias.clone().detach().reshape(bias.shape[0], 1),
                bias.clone().detach().reshape(bias.shape[0], 1),
            ],
            dim=1,
        )

        mask = []
        for i in range(self.boxes[-1].lb.shape[0]):
            if self.boxes[-1].ub[i] < 0:
                mask.append([True, False, False, False])
            elif self.boxes[-1].lb[i] > 0:
                mask.append([False, True, False, False])
            elif self.boxes[-1].ub[i] <= -self.boxes[-1].lb[i]:
                mask.append([False, False, True, False])
            else:
                mask.append([False, False, False, True])

        mask = torch.tensor(mask)

        linear_bound = LinearBound(
            torch.diag(lower_bound[mask]), torch.diag(upper_bound[mask]), lower_bias[mask], upper_bias[mask]
        )
        self.linear_bounds.append(linear_bound)

        box = self.backsubstitute(-1)
        self.boxes.append(box)

    def propagate_leaky_relu(self, leaky_relu: nn.LeakyReLU):
        negative_slope = leaky_relu.negative_slope
        box = self.boxes[-1]
        prev_lb, prev_ub = box.lb, box.ub

        ## set params for prev_lb >= 0
        non_negative = (prev_lb >= 0).float()
        U = L = torch.diag(non_negative)
        u = l = torch.zeros_like(prev_lb)
        ## prev_ub <= 0 means weights and bias are all zeros, so can leave U, L, u, l as is

        ## set params for crossing, i.e. prev_lb < 0 and prev_ub > 0
        crossing = torch.logical_and(prev_lb < 0, prev_ub > 0).float()
        lmbda = (prev_ub - negative_slope * prev_lb) / (prev_ub - prev_lb)
        b = (negative_slope - 1) * prev_lb / (prev_ub - prev_lb)
        if negative_slope <= 1:
            U += torch.diag(lmbda * crossing)  # unique
            L += torch.diag(negative_slope * crossing)  # may be optimized; in the range [negative_slope, 1]
            u += b * crossing
            # l += torch.zeros_like(b)
        else:
            U += torch.diag(crossing)  # may be optimized; in the range [1, negative_slope]
            L += torch.diag(lmbda * crossing)  # unique
            # u += torch.zeros_like(b)
            l += b * crossing

        linear_bound = LinearBound(U, L, u, l)
        self.linear_bounds.append(linear_bound)

        box = self.backsubstitute(-1)
        self.boxes.append(box)


def certify_sample(model, x, y, eps) -> bool:
    for param in model.parameters():
        param.requires_grad = False
    attach_shapes(model, x.unsqueeze(0).shape)
    box = propagate_sample(model, x, eps)
    return box.check_postcondition(y)


def propagate_sample(model, x, eps) -> Box:
    dp = DeepPoly(x, eps)
    for layer in model:
        if isinstance(layer, nn.Linear):
            dp.propagate_linear(layer)
        elif isinstance(layer, nn.Conv2d):
            dp.propagate_conv(layer)
        elif isinstance(layer, nn.Flatten):
            continue
        elif isinstance(layer, nn.ReLU):
            # dp.propagate_relu(layer)
            dp.propagate_leaky_relu(nn.LeakyReLU(0))
        elif isinstance(layer, nn.LeakyReLU):
            dp.propagate_leaky_relu(layer)
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
    return dp.boxes[-1]
