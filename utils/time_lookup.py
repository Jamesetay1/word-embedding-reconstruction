import argparse, os
import random
import numpy as np
import regex as re
import time

random.seed(0)
np.random.seed(0)


def main(args):
    # Read in replacement path and get dict
    emb_dict = {}
    with open(args.vec_path, mode='r', encoding='utf-8') as f:
        for line in f:
            emb_dict[line.split()[0]] = line.split()[1:]

    # Read and write vec, replace is word is in dict
    # Read in a vec file, write word and a random embedding
    start_time = time.time()
    with open(args.write_path, mode='w', encoding='utf-8') as g:
        for word in emb_dict.keys():
                g.write(f"{word} ")
                print(*emb_dict[word], file=g)
    print("Retrieve & Write: --- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Functionality Options
    parser.add_argument('--vec_path', dest='vec_path', type=str, default='', help='path of original vectors')
    parser.add_argument('--write_path', dest='write_path', type=str)

    # Pass args to main
    args = parser.parse_args()
    main(args)
