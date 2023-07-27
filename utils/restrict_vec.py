import argparse, os
import random
import numpy as np
import regex as re

random.seed(0)
np.random.seed(0)


def main(args):
    with open(args.read_path, mode='r', encoding='utf-8') as f:
        with open(args.write_path, mode='w', encoding='utf-8') as g:
            for line in f:
                string = line.split()[0]
                if len(string) > args.len_restrict:
                    continue
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
    parser.add_argument('--len_restrict', dest='len_restrict', type=int, default=1000)
    parser.add_argument('--drop_numbers', action='store_true', help='remove numbers as well')

    # Pass args to main
    args = parser.parse_args()
    main(args)
