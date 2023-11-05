import argparse
import torch

from networks import get_network
from utils.loading import parse_spec

from DeepPoly import certify_sample

DEVICE = "cpu"


def analyze(net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:
    return certify_sample(net, inputs, true_label, eps)


def main(net, spec):
    true_label, dataset, image, eps = parse_spec(spec)

    net = get_network(net, dataset, f"models/{dataset}_{net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        return "verified"
    else:
        return "not verified"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network verification using DeepPoly relaxation.")
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()
    print(main(args.net, args.spec))
