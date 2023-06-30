import torch
import torch.nn as nn
import numpy as np
import json, sys, argparse, os, math
from torch.nn.utils.rnn import pad_sequence
import train
from train import pad_collate

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, vec_path, npy_path):
        self.vec_path = vec_path
        self.npy_path = npy_path
        self.data_list = np.load(self.npy_path, allow_pickle=True)

        pass
    def __getitem__(self, index):

        self.data_item = self.data_list[index]

        word = nn.functional.one_hot(torch.IntTensor(self.data_item[0]).long(), num_classes=54)
        word = word.to(torch.float)
        vec = torch.FloatTensor(self.data_item[1])

        return word, vec

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.data_list)


def main(args):
    p = nn.ParameterDict({})
    # Embedding and Encoding parameters
    p['vocab_size'] = 55  # 54 chars, + all zeroes is padding
    p['embed_dim'] = 300

    # Learning Hyperparameters
    p['epochs'] = 300
    p['batch_size'] = args.batch_size
    p['learning_rate'] = 0.0001

    # Filter and Channel Parameters
    p['filter_widths'] = [1, 2, 3, 4, 5, 6, 7]
    p['intervals'] = [4, 4, 4, 4, 5, 6, 7]
    p['filter_interval'] = args.filter_interval
    p['num_filters'] = list(map(lambda x: x * p['filter_interval'], p['intervals']))
    p['final_linear_width'] = sum(p['num_filters'])

    print("\nStarting reconstruction with the following parameters:")
    for k, v in p.items():
        print(str(k) + ': ' + str(v))
    print("\n")

    # Get a char to id & id to char mapping dicts
    char_to_id = get_char_dict(args.charset_path)
    id_to_char = {v: k for k, v in char_to_id.items()}
    print(id_to_char)
    # Load in test dataset
    test_dataset = train.CustomDataset(args.testset_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p['batch_size'], collate_fn=pad_collate, shuffle=False)

    # Load in model
    model = train.ConvNet(p).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()


    # Go through each word in the test set and reconstruct its embedding
    reconstructed_dict = {}
    for i, (word_oh, vec) in enumerate(test_loader):
        word_oh = word_oh.to(device)
        #vec = vec.to(device)

        # Do a prediction!
        outputs = model(word_oh)

        # Convert back to character form
        for item in range(word_oh.size(dim=0)):
            word_ids = torch.argmax(word_oh[item], dim=1).tolist()
            word_ids = [i for i in word_ids if i != 0]  # Padding
            print(word_ids)
            chars = list([*map(id_to_char.get, word_ids)])
            print(chars)
            word = ''.join(chars)
            reconstructed_dict[word] = outputs[item]

    print(f'reconstructed path: {args.reconstructed_path}')
    with open(args.reconstructed_path, mode='w', encoding='utf-8') as f:
        for word, vec in reconstructed_dict.items():
            vec = list(np.around(np.array(vec.tolist()), 4))

            f.write(f'{word} ')
            print(*vec, file=f)



    print(id_to_char)

def get_char_dict(charset_path):
    char_dict = {}
    with open(charset_path, mode="r", encoding='utf-8') as f:
        for char_set in f:
            for i, c in enumerate(char_set):
                char_dict[c] = i+1

    return char_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Functionality Options
    parser.add_argument('--model_path', dest='model_path', type=str, help='log path in the format epoch,trainloss,devloss\\n')
    parser.add_argument('--testset_path', dest='testset_path', type=str, help='set of preprocessed word/vec pairs in .npy format')
    parser.add_argument('--reconstructed_path', dest='reconstructed_path', type=str, help='location of where to write reconstructed .vec file')
    parser.add_argument('--charset_path', dest='charset_path', type=str, help='charset to use for mapping back to words')

    parser.add_argument('--filter_interval', dest='filter_interval', type=int, default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)

    # Inference
    args = parser.parse_args()
    main(args)