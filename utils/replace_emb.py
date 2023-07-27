import argparse, os
import random
import numpy as np
import regex as re

random.seed(0)
np.random.seed(0)


def main(args):
    # Read in replacement path and get dict
    replacement_dict = {}
    with open(args.rep_path, mode='r', encoding='utf-8') as f:
        for line in f:
            replacement_dict[line.split()[0]] = line.split()[1:]

    # Read and write vec, replace is word is in dict
    # Read in a vec file, write word and a random embedding
    with open(args.vec_path, mode='r', encoding='utf-8') as f:
        with open(args.write_path, mode='w', encoding='utf-8') as g:
            for line in f:
                word = line.split()[0]
                if word in replacement_dict:
                    g.write(f"{word} ")
                    print(*replacement_dict[word], file=g)
                else:
                    g.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Functionality Options
    parser.add_argument('--vec_path', dest='vec_path', type=str, default='', help='path of original vectors')
    parser.add_argument('--rep_path', dest='rep_path', type=str, default='', help='path of reference vectors')
    parser.add_argument('--write_path', dest='write_path', type=str)

    # Pass args to main
    args = parser.parse_args()
    main(args)
