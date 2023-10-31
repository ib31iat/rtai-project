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

    def propagate_linear(self, fc: nn.Linear) -> 'AbstractBox':
        # TODO: Something is wrong here, fc_6 for example results in a shape problem in center@fc.weight.t()
        assert self.lb.shape == self.ub.shape
        center = (self.lb + self.ub) / 2
        eps = (self.ub - self.lb) / 2

        center_out = center@fc.weight.t()
        if fc.bias is not None:
            center_out = center_out + fc.bias
        eps_out = eps@fc.weight.abs().t()
        lb = center_out - eps_out
        ub = center_out + eps_out
        return AbstractBox(lb, ub)

    def propagate_relu(self, relu: nn.ReLU) -> 'AbstractBox':
        lb = relu(self.lb)
        ub = relu(self.ub)
        return AbstractBox(lb, ub)

    def propagate_leaky_relu(self, relu: nn.LeakyReLU) -> 'AbstractBox':
        # TODO: Implement leaky ReLU
        lb = relu(self.lb)
        ub = relu(self.ub)
        return AbstractBox(lb, ub)

    def propagate_flatten(self, flatten: nn.Flatten) -> 'AbstractBox':
        # TODO: Check if that is really correct, I suspect there is something wrong here
        lb = flatten(self.lb)
        ub = flatten(self.ub)
        return AbstractBox(lb, ub)

    def propagate_conv2d(self, conv: nn.Conv2d):
         # TODO: Implement Conv layer
        return AbstractBox(self.lb, self.ub)

    def check_postcondition(self, y) -> bool:
        target = y
        target_lb = self.lb[0][target].item()
        for i in range(self.ub.shape[-1]):
            if i != target and self.ub[0][i] >= target_lb:
                return False
        return True

def certify_sample(model, x, y, eps) -> bool:
    box = propagate_sample(model, x, eps)
    return box.check_postcondition(y)

def propagate_sample(model, x, eps) -> AbstractBox:
    box = AbstractBox.construct_initial_box(x, eps)
    for layer in model:
        if isinstance(layer, nn.Linear):
            box = box.propagate_linear(layer)
        elif isinstance(layer, nn.ReLU):
            box = box.propagate_relu(layer)
        elif isinstance(layer, nn.Flatten):
            box = box.propagate_flatten(layer)
        elif isinstance(layer, nn.LeakyReLU):
            box = box.propagate_leaky_relu(layer)
        elif isinstance(layer, nn.Conv2d):
            box = box.propagate_conv2d(layer)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
    return box
