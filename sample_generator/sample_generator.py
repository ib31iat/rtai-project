import argparse

import os
import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np
import random as rnd

import torch
from torch import nn
from torchvision import datasets, transforms

sys.path.insert(1, "code/")
from networks import get_network


def load_datasets():
    """
    Loads the mnist and the cifar10 dataset
    """
    mnist_dataset = datasets.MNIST(  # entries (10k) in dataset: image ([1,28,28]), label
        "sample_generator/mnist_data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    cifar10_dataset = datasets.CIFAR10(  # entries (10k) in dataset: image ([3,32,32]), label
        "sample_generator/cifar10_data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    return mnist_dataset, cifar10_dataset


def fgsm_untargeted(model, x, label, eps, **kwargs):
    """
    Searches for an image that has a different label than the original image

    Args:
      x: input image
      label: true label of x
      eps: size of l-infinity ball
    """
    x_ = x.clone().detach_()
    x_.requires_grad_()
    model_out = model(x_)
    label = torch.LongTensor([label])
    model.zero_grad()  # zero out gradient to not be influenced by previous iterations
    loss = nn.CrossEntropyLoss()(model_out, label)
    loss.backward()  # gradient computation

    res = x_ + eps * x_.grad.sign()

    return res


def pgd(model, x, label, k, eps, eps_step, model_to_prob, clip_min=0, clip_max=1):
    """
    Does many iteration of the untargeted fgsm to find an image with a different label

    Args:
      x: input image
      label: true label of x
      k: number of FGSM iterations
      eps: size of l-infinity ball
      eps_step: step size of FGSM iterations
    """
    x_projected = x + eps * (2 * torch.rand_like(x) - 1)
    x_projected.clamp_(min=clip_min, max=clip_max)

    for i in tqdm(range(k), leave=False):
        x_prime = fgsm_untargeted(model, x_projected, label, eps_step)
        x_projected = x_prime.clamp(x - eps, x + eps)

        if model_to_prob(x_projected).detach().numpy().argmax() != label:
            return i, "not verified", x_projected

    x_projected.clamp_(min=clip_min, max=clip_max)

    return i, "verified", x_projected


def generate_sample(net, mnist_dataset, cifar10_dataset):
    """
    Generates a sample and stores whether it is verified or not

    Args:
      net: the neural network for which the sample should be generated
      mnist_dataset: the mnist dataset
      cifar10_dataset: the cifar10 dataset
    """
    output_folder = "test_cases_additional"
    ground_truth_file = f"{output_folder}/gt.txt"

    nets = [
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
    ]

    if net is None:
        net = nets[rnd.randint(0, len(nets) - 1)]

    if net == "conv_3":
        dataset_name = "cifar10"
    elif net == "fc_6":
        dataset_name = "mnist" if rnd.randint(0, 1) else "cifar10"
    else:
        dataset_name = "mnist"

    dataset = mnist_dataset if dataset_name == "mnist" else cifar10_dataset
    image_idx = rnd.randint(0, len(dataset))
    image, true_label = dataset[image_idx]
    image = image.unsqueeze(0)
    eps = round(rnd.uniform(0, 0.3), 4)
    k = int(1e4)
    pgd_iterations = int(1e1)
    eps_step = 2.5 * (eps - 1e-6) / k

    model = get_network(net, dataset_name, f"models/{dataset_name}_{net}.pt").to("cpu")
    model_to_prob = nn.Sequential(model, nn.Softmax())

    # Check if the model can predict the image correctly
    # mnist_1709 is for example not correctly predicted by fc_base
    predicted_label = model(image).max(dim=1)[1].item()

    if predicted_label != true_label:
        return False

    pgd_fail = 0
    for i in tqdm(range(pgd_iterations), leave=False):
        pgd_fail = i
        faile_idx, result, _ = pgd(
            model, image, label=true_label, k=k, eps=eps, eps_step=eps_step, model_to_prob=model_to_prob
        )
        if result == "not verified":
            break

    image_file_name = f"img{image_idx}_{dataset_name}_{eps}.txt"
    image_full_path = f"{output_folder}/{net}/{image_file_name}"

    file = Path(image_full_path)
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(image_full_path, "w") as output_file:
        output_file.write(f"{true_label}\n")
        image_list = list(image.flatten().numpy())
        output_file.write(f'{",".join(f"{pixel:.16f}".rstrip("0") for pixel in image_list)}')
        output_file.write("\n")

    with open(ground_truth_file, "a") as gt_file:
        gt_file.write(f"{net},{image_file_name},{result}\n")

    if result == "not verified":
        with open(f"{output_folder}/faile_idx.txt", "a") as file:
            file.write(f"{image_file_name[:-4]}: {pgd_fail}, {faile_idx}\n")

    return True


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Create additional training samples")
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
        default=None,
        help="Neural network architecture for which a sample should be generated",
    )
    parser.add_argument(
        "-n",
        "--number-of-samples",
        type=int,
        default=1,
        help="The number of samples that should be generated",
    )

    args = parser.parse_args()

    # Load the datasets
    mnist_dataset, cifar10_dataset = load_datasets()

    # generate samples
    for _ in tqdm(range(args.number_of_samples)):
        successful = False
        while not successful:
            # Retry in case the selected model cannot classify the image correctly
            # to still get the demanded number of samples.
            successful = generate_sample(args.net, mnist_dataset, cifar10_dataset)


if __name__ == "__main__":
    # Start main function
    main()
