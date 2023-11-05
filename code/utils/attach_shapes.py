import numpy as np
import torch
import torch.nn as nn


def attach_shapes(model, input_size):
    """
    Add input and output shape attributes to each module.
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

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)
    x = torch.rand(input_size)
    model(x)
    for h in hooks:
        h.remove()


# from networks import get_network

# net = get_network("conv_4", "cifar10")
# shapes = attach_shapes(net, (1, 3, 32, 32))
# for module in net.modules():
#     if type(module) != nn.Sequential:
#         print(module.input_shape, module.output_shape, module.in_features, module.out_features)
