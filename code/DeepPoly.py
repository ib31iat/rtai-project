from dataclasses import dataclass
from time import time
from typing import Dict, List, Optional, Union

import torch
from torch.autograd.functional import jacobian
from torch.nn.functional import relu, sigmoid
import torch.nn as nn

from Box import Box
from utils.utils import has_relu, preprocess_net


@dataclass(frozen=True)
class LinearBound:
    lower_weight: torch.Tensor
    upper_weight: torch.Tensor
    lower_bias: Optional[torch.Tensor] = None
    upper_bias: Optional[torch.Tensor] = None

    def get_params(self):
        return self.lower_weight.clone(), self.upper_weight.clone(), self.lower_bias.clone(), self.upper_bias.clone()


class DeepPoly:
    def __init__(self, model: nn.Module, x: torch.Tensor, y: int, eps: float):
        self.initial_box = Box.construct_initial_box(x, eps)
        # a list of Boxes; each element stores the concrete bounds for one layer
        self.boxes: List[Box] = []
        # a list of LinearBounds; each element stores the linear bounds for one layer
        self.linear_bounds: List[LinearBound] = []
        # If the bounds are stored as above, then every propagate method
        # should append a box and a linear bound to the lists above.

        self.model = model
        # stores the slope parameters for the ReLU layers
        self.alphas: Dict[int, torch.Tensor] = self._initial_alphas()  # {layer_number: alpha}

    def _initial_alphas(self) -> dict:
        """
        Initializes the alphas for the ReLU layers (to 0). Note that the alphas aren't the actual slopes
        (cf. propagate_relu)
        """
        alphas = {}  # {layer_number: alpha}
        for i, layer in enumerate(self.model):
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                # initial_alpha = torch.zeros(layer.in_features)
                # initial_alpha = torch.rand(layer.in_features) * 4 - 2  # uniform in [-2, 2]
                initial_alpha = -2 * torch.ones(layer.in_features)
                alphas[i] = torch.nn.Parameter(initial_alpha)

        return alphas

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
                    lower_bias = torch.zeros(lower_weight.shape[0])
                    upper_bias = torch.zeros(upper_weight.shape[0])
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

    def propagate_relu(self, relu: Union[nn.LeakyReLU, nn.ReLU], layer_number: int):
        slope = relu.negative_slope
        box = self.boxes[-1]
        prev_lb, prev_ub = box.lb, box.ub

        ## set bounds for prev_lb >= 0 and prev_ub <= 0
        L_diag = 1.0 * (prev_lb >= 0) + slope * (prev_ub <= 0)
        U_diag = 1.0 * (prev_lb >= 0) + slope * (prev_ub <= 0)
        l = torch.zeros_like(prev_lb)
        u = torch.zeros_like(prev_ub)

        ## set bounds for crossing, i.e. prev_lb < 0 < prev_ub
        crossing_selector = torch.logical_and(prev_lb < 0, prev_ub > 0).float()
        lmbda = (prev_ub - slope * prev_lb) / (prev_ub - prev_lb)
        b = (slope - 1) * prev_lb * prev_ub / (prev_ub - prev_lb)
        alpha = self.alphas[layer_number]
        if slope <= 1:
            bound_slope = sigmoid(alpha) * (1 - slope) + slope  # slope needs to be in [slope, 1]
            U_diag += lmbda * crossing_selector  # tightest possible linear upper bound
            L_diag += bound_slope * crossing_selector
            u += b * crossing_selector
            # l += torch.zeros_like(b)
        else:
            bound_slope = sigmoid(alpha) * (slope - 1) + 1  # slope needs to be in [1, slope]
            U_diag += bound_slope * crossing_selector
            L_diag += lmbda * crossing_selector  # tightest possible linear lower bound
            # u += torch.zeros_like(b)
            l += b * crossing_selector

        L = torch.diag(L_diag)
        U = torch.diag(U_diag)
        linear_bound = LinearBound(L, U, l, u)
        self.linear_bounds.append(linear_bound)

        # no need to backsubstitute as concrete bounds are only used in relu propagation; relu is never followed by relu
        # box = self.backsubstitute(-1)
        # self.boxes.append(box)

    def propagate(self) -> Box:
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                self.propagate_linear(layer)
            elif isinstance(layer, nn.Conv2d):
                self.propagate_conv(layer)
            elif isinstance(layer, nn.Flatten):
                continue
            elif isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                self.propagate_relu(layer, layer_number=i)
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
        return self.boxes[-1]

    def flush(self):
        self.boxes = []
        self.linear_bounds = []

    def optimize(self, y: int) -> bool:
        params = self.alphas.values()
        optimizer = torch.optim.Adam(params, lr=0.1)

        start_time = time()
        while time() - start_time < 60:  # limit verification to 60 seconds
            optimizer.zero_grad()
            box = self.propagate()
            if box.check_postcondition(y):
                return True
            lb, ub = box.lb, box.ub
            pairwise_difference = lb[y] - ub
            pairwise_difference[y] = 0
            loss = relu(-pairwise_difference).sum()
            loss.backward()
            optimizer.step()
            self.flush()

    def verify(self, y: int) -> bool:
        if has_relu(self.model):
            return self.optimize(y)
        else:
            box = self.propagate()
            return box.check_postcondition(y)


def certify_sample(model, x, y, eps) -> bool:
    preprocess_net(model, x.unsqueeze(0).shape)
    dp = DeepPoly(model, x, y, eps)
    return dp.verify(y)
