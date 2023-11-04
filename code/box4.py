import torch
import torch.nn as nn

class AbstractBox:
    # TODO: adapt for back sub

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor, lb_abstract=None, ub_abstract=None, lb_bias=None, ub_bias=None):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub
        self.lb_abstract = lb_abstract
        self.ub_abstract = ub_abstract
        self.lb_bias = lb_bias
        self.ub_bias = ub_bias

    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'AbstractBox':
        lb = x - eps
        lb.clamp_(min=0, max=1)

        ub = x + eps
        ub.clamp_(min=0, max=1)

        return AbstractBox(lb, ub)

    def propagate_linear(self, fc: nn.Linear) -> 'AbstractBox':
        assert self.lb.shape == self.ub.shape
        center = (self.lb + self.ub) / 2
        eps = (self.ub - self.lb) / 2

        center_out = center@fc.weight.t()
        if fc.bias is not None:
            center_out = center_out + fc.bias
        eps_out = eps@fc.weight.abs().t()
        lb = center_out - eps_out
        ub = center_out + eps_out
        return AbstractBox(lb, ub, fc.weight, fc.weight, fc.bias, fc.bias)

    def propagate_relu(self, relu: nn.ReLU) -> 'AbstractBox':
        lb = relu(self.lb)
        ub = relu(self.ub)
        return AbstractBox(lb, ub)

    def propagate_leaky_relu(self, relu: nn.LeakyReLU) -> 'AbstractBox':
        # TODO: Implement leaky ReLU
        lb = relu(self.lb)
        ub = relu(self.ub)
        return AbstractBox(lb, ub)

    def propagate_flatten(self, flatten: nn.Flatten, in_dim, out_dim) -> 'AbstractBox':
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
    box_back_sub = back_substitution(model, box)
    return box_back_sub.check_postcondition(y)

def propagate_sample(model, x, eps) -> AbstractBox:
    box = [] # to store intermediate boxes for back substitution
    box.append(AbstractBox.construct_initial_box(x, eps))
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            box.append(box[-1].propagate_linear(layer))
        elif isinstance(layer, nn.ReLU):
            box.append(box[-1].propagate_relu(layer))
        elif isinstance(layer, nn.Flatten):
            if i-1 < 0:
                in_dim = x.shape
            else:
                out_dim = model[i-1].out_features
            out_dim = model[i+1].in_features
            box.append(box[-1].propagate_flatten(layer, in_dim, out_dim))
        elif isinstance(layer, nn.LeakyReLU):
            box.append(box[-1].propagate_leaky_relu(layer))
        elif isinstance(layer, nn.Conv2d):
            box.append(box[-1].propagate_conv2d(layer))
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
    return box

def back_substitution(model, box_list):
    box = AbstractBox(box_list[-1].lb, box_list[-1].ub, box_list[-1].lb_abstract, box_list[-1].ub_abstract, box_list[-1].lb_bias, box_list[-1].ub_bias)
    for i, layer in enumerate(reversed(model)):
        if i == 0:
            continue
        # TODO: Upper lower bound distinction
        if isinstance(layer, nn.Linear):
            # weigths_lb = box_list[-(i+1)].lb_abstract
            # weigths_ub = box_list[-(i+1)].ub_abstract

            # lb = box.lb_abstract
            # ub = box.ub_abstract

            # box.lb_abstract = torch.maximum(torch.tensor(0), lb) @ weigths_lb+ torch.minimum(torch.tensor(0), lb) @ weigths_ub

            # box.ub_abstract = torch.maximum(torch.tensor(0), ub) @ weigths_ub + torch.minimum(torch.tensor(0), ub) @ weigths_ub

            box.lb_abstract, box.ub_abstract, box.lb_bias, box.ub_bias = multiplication(box, box_list[-(i+1)])
        elif isinstance(layer, nn.Flatten):
            pass
            # box.lb_abstract = box.lb_abstract.reshape(10,28,28)
            # box.ub_abstract = box.ub_abstract.reshape(10,28,28)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

        # if len(box.lb_abstract.shape) == 2:
        #     print(box.lb_abstract.shape)
        #     print(box.lb_abstract[:,-1])

    lb, ub, lb_bias, ub_bias = multiplication(box, box_list[0], final=True)
    # lb = torch.sum(lb, dim=(1,2)) + lb_bias
    # ub  = torch.sum(ub, dim=(1,2)) + ub_bias
    lb = lb +lb_bias
    ub = ub + ub_bias

    box.lb = lb.reshape(1, lb.shape[0])
    box.ub = ub.reshape(1, ub.shape[0])
    return box

def multiplication(bounds, previous_bounds, final=False): # one layer later, current
    lb = bounds.lb_abstract
    ub = bounds.ub_abstract

    if final:
        weigths_lb = previous_bounds.lb.flatten()
        weigths_ub = previous_bounds.ub.flatten()

        # lb_new = torch.maximum(torch.tensor(0), lb) * weigths_lb+ torch.minimum(torch.tensor(0), lb) * weigths_ub

        # ub_new = torch.maximum(torch.tensor(0), ub) * weigths_ub + torch.minimum(torch.tensor(0), ub) * weigths_ub
    else:
        weigths_lb = previous_bounds.lb_abstract
        weigths_ub = previous_bounds.ub_abstract

    lb_new = torch.maximum(torch.tensor(0), lb) @ weigths_lb+ torch.minimum(torch.tensor(0), lb) @ weigths_ub

    ub_new = torch.maximum(torch.tensor(0), ub) @ weigths_ub + torch.minimum(torch.tensor(0), ub) @ weigths_ub

    if previous_bounds.lb_bias is None:
        lb_bias = bounds.lb_bias
        ub_bias = bounds.ub_bias
    else:
        lb_bias = torch.maximum(torch.tensor(0), lb) @ previous_bounds.lb_bias + torch.minimum(torch.tensor(0), lb) @ previous_bounds.ub_bias + bounds.lb_bias

        ub_bias = torch.maximum(torch.tensor(0), ub) @ previous_bounds.ub_bias + torch.minimum(torch.tensor(0), ub) @ previous_bounds.lb_bias + bounds.ub_bias

    return lb_new, ub_new, lb_bias, ub_bias
