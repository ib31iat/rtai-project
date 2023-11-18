import argparse
import os
from tqdm import tqdm

from verifier import main as verifier


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Check with ground truths")

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
        nargs="+",  # to validate mulitple nets
        default=[
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
        ],  # if none specified, check all nets
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument(
        "--additional",
        action='store_true'
    )
    args = parser.parse_args()

    # Set the base folder
    if args.additional:
        base_path = "test_cases_additional"
    else:
        base_path = "test_cases"

    # Read ground truths
    ground_truth = {}  # {net: {spec: verified | not verified}}
    with open(f"{base_path}/gt.txt", "r") as ground_truth_file:
        for line in ground_truth_file:
            split_line = line.split(",")
            net = split_line[0]
            spec = split_line[1]
            state = split_line[2].replace("\n", "")
            ground_truth.setdefault(net, {})[spec] = state

    # Check supplied nets
    failures = []
    pbar = tqdm(args.net)
    for net in pbar:
        pbar.set_description(f"{net}")
        pbar_inner = tqdm(os.listdir(f"{base_path}/{net}"), leave=False)
        for spec in pbar_inner:
            pbar_inner.set_description(f"{spec[:-4]}")
            result = verifier(net, f"{base_path}/{net}/{spec}")
            gt = ground_truth[net][spec]
            if result != gt:
                failures.append(f"net: {net}, spec: {spec}. Was '{result}' should  have been '{gt}'")

    # Print
    if len(failures) == 0:
        print(f"All test cases are correct. Tested nets: {', '.join(args.net)}")
    else:
        print("There were some errors:")
        for f in failures:
            print(f"\t- {f}")


if __name__ == "__main__":
    # Start main function
    main()
