import argparse, os
import random
import numpy as np
import regex as re

random.seed(0)
np.random.seed(0)


def main(args):
    stored_dict = {}
    repetitions_skipped = 0

    # Read in a vec file, write word and a random embedding
    with open(args.read_path, mode='r', encoding='utf-8') as f:
        with open(args.write_path, mode='w', encoding='utf-8') as g:
            for line in f:
                word = line.split()[0]
                vec = []
                for i in range (1, 300):
                    vec.append(random.uniform(-1, 1))

                g.write(f"{word} ")
                print(*vec, file=g)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Functionality Options
    parser.add_argument('--read_path', dest='read_path', type=str, default='', help='where to write the results file')
    parser.add_argument('--write_path', dest='write_path', type=str, default='', help='where to write the results file')

    # Pass args to main
    args = parser.parse_args()
    main(args)
