import json, sys, argparse, os
import numpy as np


def word_to_id_list(word_list, char_dict):

    id_list = []
    for word in word_list:
        char_list=[]
        for c in word:
            char_list.append(char_dict[c])

        id_list.append(char_list)

    return id_list


def main(args):


    # Define character dict from file
    char_dict = {}
    with open(args.charset_path, mode="r", encoding='utf-8') as f:
        for char_set in f:
            for i, c in enumerate(char_set):
                char_dict[c] = i+1

    print(char_dict)
    word_list = []
    vec_list = []

    with open(args.vec_path, mode="r", encoding='utf-8') as f:
        for i, line in enumerate(f):

            word_list.append(line.split()[0])
            vec_list.append(list(map(float, line.split()[1:])))

    id_list = word_to_id_list(word_list, char_dict)
    data_list = list(zip(id_list, vec_list))

    np_list = np.array(data_list, dtype=object)
    np.save(args.npy_path, np_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Functionality Options
    parser.add_argument('--vec_path', dest='vec_path', type=str, help='path to vector file to create map of')
    parser.add_argument('--charset_path', dest='charset_path', type=str, help='path to charset file')
    parser.add_argument('--npy_path', dest='npy_path', type=str)

    args = parser.parse_args()
    main(args)