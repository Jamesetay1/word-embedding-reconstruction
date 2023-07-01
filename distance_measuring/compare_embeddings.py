import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import argparse
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree

def main(args):
    # Read in arguments
    rec_dict, rec_vecs = read_vec(args.reconstructed_path, args.top_n)
    ref_dict, ref_vecs = read_vec(args.reference_path, args.top_n)
    ref_words = list(ref_dict.keys())

    print(f'rec_dict created and of size {len(rec_dict)} with vec matrix of shape {rec_vecs.shape}')
    print(f'ref_dict created and of size {len(ref_dict)} with vec matrix of shape {ref_vecs.shape}')


    # Get cosine similarity vector
    if args.cossim:
        print("Doing the cosine similarity now")
        cossim_vector = np.diagonal(cosine_similarity(rec_vecs, ref_vecs))

    # Get n closest
    if args.nearest > 0:
        print(f"Doing nn check with nn = {args.nearest}")

        # Set up tree
        vec_tree = KDTree(ref_vecs)
        all_neighbors = (vec_tree.query(rec_vecs, k=args.nearest)[1]).tolist()
        print(all_neighbors)
        neighbors_dict = defaultdict(list)
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                neighbors_dict[ref_words[i]].append(ref_words[neighbor])


    print(neighbors_dict)

    # Write to output file
    ref_name = "ref-" + os.path.split(args.reconstructed_path)[1].split('.')[0]
    rec_name = "rec-" + os.path.split(args.reference_path)[1].split('.')[0]
    results_path = f"{args.output_dir}{rec_name}_{ref_name}.txt"
    print(f"writing to {args.output_dir}")

    with open(results_path, mode="w", encoding="utf-8") as f:
        f.write(f"Comparison between: reference embeddings <{args.reference_path}> and reconstructed embeddings <{args.reconstructed_path}>\n")
        f.write(f"====================================================\n")
        f.write(f"mean cosine similarity: {np.mean(cossim_vector)}\n")
        f.write(f"mode cosine similarity: {stats.mode(cossim_vector, keepdims=True)[0]}\n")
        f.write(f"====================================================\n")
        f.write(f"Precision at {args.nearest} = ")

def read_vec(path, top_n):

    word_dic = {}
    # Create a dictionary and fill it up - word:vec
    with open(path, mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            if i == top_n:
                break

            word = line.split()[0]
            vec = line.split()[1:]
            word_dic[word] = [float(j) for j in vec]

    return word_dic, np.array(list(word_dic.values()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path for reconstructed and reference paths
    parser.add_argument('--reconstructed_path', type=str, help='log path in the format epoch,trainloss,devloss\\n')
    parser.add_argument('--reference_path', type=str)

    parser.add_argument('--top_n', type=int, default=1000)

    # Comparison options
    parser.add_argument('--cossim', action='store_true')
    parser.add_argument('--nearest', type=int, default=0)
    parser.add_argument('--wordsim', action='store_true')


    # Output path
    parser.add_argument('--output_dir', type=str, default='results/reconstructed_embeddings/')


    args = parser.parse_args()
    main(args)