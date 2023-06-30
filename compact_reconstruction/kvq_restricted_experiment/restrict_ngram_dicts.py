import json, sys, argparse, os
from datetime import datetime

HOME = os.getenv("HOME")
import random
import numpy as np
import regex as re

random.seed(0)
np.random.seed(0)


def main(args):
    with open(args.read_path, mode='r') as f:
        with open(args.write_path, mode='w') as g:
            for line in f:
                string = line.split()[0]
                if re.search("[^a-z0-9öüé\'-.+&°€$%£:]", string) is None:
                    if args.drop_numbers:
                        g.write(f'{string}\n')
                    else:
                        g.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Functionality Options
    parser.add_argument('--read_path', dest='read_path', type=str, default='', help='where to write the results file')
    parser.add_argument('--write_path', dest='write_path', type=str, default='', help='where to write the results file')
    parser.add_argument('--drop_numbers', action='store_true')
    # Pass args to main
    args = parser.parse_args()
    main(args)
