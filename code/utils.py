import torch
import torch.nn as nn

from networks import dln_conv_model, dln_model, fc_model, conv_model


def attach_shapes(model, input_size):
    """
    Add input and output shape attributes to each module.
    """

    def register_hook(module):
        def hook(module, input, output):
            input_shape = list(input[0].size())[1:]
            module.input_shape = input_shape
            if isinstance(output, (list, tuple)):
                output_shape = [[-1] + list(o.size())[1:] for o in output]
            else:
                output_shape = list(output.size())[1:]
            module.output_shape = output_shape

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
#         print(module.input_shape, module.output_shape)
