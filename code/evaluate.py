import argparse
import subprocess
import os

def main():
  # Create argument parser
  parser = argparse.ArgumentParser(
    description="Check with ground truths"
  )

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
    nargs='+', # to validate mulitple nets
    required=True,
    help="Neural network architecture which is supposed to be verified.",
  )
  args = parser.parse_args()

  # Read ground truths
  ground_truth = {} # {net: {spec: verified | not verified}}
  with open('test_cases/gt.txt', 'r') as ground_truth_file:
    for line in ground_truth_file:
      split_line = line.split(',')
      net = split_line[0]
      spec = split_line[1]
      state = split_line[2].replace('\n', '')
      ground_truth.setdefault(net, {})[spec] = state

  # Check supplied nets
  failures = []
  for net in args.net:
    for spec in os.listdir('test_cases/{}'.format(net)):
      command = 'python3 code/verifier.py --net {0} --spec test_cases/{0}/{1}'.format(net, spec)
      result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
      command_error = result.stderr
      if command_error != '':
        print(command_error)
        exit()
      parsed_result = result.stdout.replace('\n','')
      gt = ground_truth[net][spec]
      if parsed_result != gt:
        failures.append('net: {}, spec: {}. Was \'{}\' should  have been \'{}\''.format(net, spec, parsed_result, gt))

  # Print
  if len(failures) == 0:
    print('All test cases are correct. Tested nets: {}'.format(*args.net, sep=','))
  else:
    print('There were some errors:')
    for f in failures:
      print('\t- {}'.format(f))

if __name__ == "__main__":
  # Start main function
  main()
