import torch.nn as nn


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


class View(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def get_model() -> nn.Sequential:
    # Add the data normalization as a first "layer" to the network
    # this allows us to search for adversarial examples to the real image, rather than
    # to the normalized image
    net = nn.Sequential(
        nn.Linear(28 * 28, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    return nn.Sequential(Normalize(), View(), *net)
