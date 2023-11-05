# Evaluate.py
This script can be used to test the Deep Poly Approximation and automatically verify the with the ground truth in the file `gt.txt`. Make sure to run the script from the root folder of the project to have all file paths correct. To run the script you need to specify at least one net that should be evaluated.
The script then outputs whether all tests where successful or not. If not, it outputs the test cases that were wrong. In case the that verifier returns an error the script aborts and prints the respective error message

## Invocation examples:
- `python3 code/evaluate.py --net fc_base`
- `python3 code/evaluate.py --net fc_base fc_1`

## Requirements
In addition to the to requirements of the project, the package `tqdm` needs to be installed. This package provides a progress bar.
