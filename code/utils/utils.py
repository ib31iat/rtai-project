import numpy as np
import torch
import torch.nn as nn


def preprocess_net(model, input_size):
    for param in model.parameters():
        param.requires_grad = False
    attach_attributes(model, input_size)


def attach_attributes(model, input_size):
    """
    Adds in-/output shape, in-/out feature number attributes to each module
    and adds negative_slope attribute to ReLU modules.
    """

    def register_hook(module):
        def hook(module, input, output):
            input_shape = list(input[0].size())
            module.input_shape = input_shape
            if not hasattr(module, "in_features"):
                module.in_features = np.prod(input_shape[1:])  # [1:] to skip batch dimension
            if isinstance(output, (list, tuple)):
                output_shape = [[-1] + list(o.size()) for o in output]
            else:
                output_shape = list(output.size())
            module.output_shape = output_shape
            if not hasattr(module, "out_features"):
                module.out_features = np.prod(output_shape[1:])  # [1:] to skip batch dimension
            if isinstance(module, nn.ReLU):
                module.negative_slope = 0.0

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)
    x = torch.rand(input_size)
    model(x)
    for h in hooks:
        h.remove()
