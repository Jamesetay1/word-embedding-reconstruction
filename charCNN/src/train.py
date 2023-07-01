import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
import json, sys, argparse, os, math
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Device configuration
print(torch.cuda.device_count())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path):
        self.npy_path = npy_path
        self.data_list = np.load(self.npy_path, allow_pickle=True)

        pass
    def __getitem__(self, index):

        self.data_item = self.data_list[index]

        word = nn.functional.one_hot(torch.IntTensor(self.data_item[0]).long(), num_classes=55)
        word = word.to(torch.float)
        vec = torch.FloatTensor(self.data_item[1])

        return word, vec

    def __len__(self):
        return len(self.data_list)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, p):
        super(ConvNet, self).__init__()

        # Extract some params
        self.V = p['vocab_size']
        self.embed_dim = p['embed_dim']

        self.num_filters = p['num_filters']
        self.filter_widths = p['filter_widths']
        self.final_linear_width = p['final_linear_width']

        self.conv_list = nn.ModuleList([
            nn.Conv1d(self.V, self.num_filters[0], kernel_size=1, padding=0),
            nn.Conv1d(self.V, self.num_filters[1], kernel_size=2, padding=0),
            nn.Conv1d(self.V, self.num_filters[2], kernel_size=3, padding=1),
            nn.Conv1d(self.V, self.num_filters[3], kernel_size=4, padding=1),
            nn.Conv1d(self.V, self.num_filters[4], kernel_size=5, padding=2),
            nn.Conv1d(self.V, self.num_filters[5], kernel_size=6, padding=2),
            nn.Conv1d(self.V, self.num_filters[6], kernel_size=7, padding=3)
        ])

        self.post_convolution = nn.Sequential(
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )

        self.sigmoid_gate = nn.Sequential(
            nn.Linear(self.final_linear_width, self.final_linear_width, bias=True),
            nn.Sigmoid()
        )

        self.tanh_transform = nn.Sequential(
            nn.Linear(self.final_linear_width, self.final_linear_width, bias=True),
            nn.Tanh()
        )

        self.linear_layer = nn.Sequential(nn.Linear(self.final_linear_width, self.embed_dim, bias=True))


    def forward(self, x):
        #print(f"before reshape: {type(x)}, {x.shape}, {x.dtype}, {x.device}")
        cur_batch_size = x.size(dim=0)
        # num_samples = x.size(dim=1)
        # vocab_size = x.size(dim=2)

        # Can't quite figure out how the transpose1D works
        x = torch.transpose(x, 1, 2)
        #print(f"after transpose: {type(x)}, {x.shape}, {x.dtype}, {x.device}")

        outputs = None
        for filter_width in self.filter_widths:
            e = self.conv_list[filter_width-1](x)
            #print(f"\nafter convolution of size {filter_width}, before tanh: {type(e)}, {e.shape}, {e.dtype}, {e.device}")

            # Undo transpose after convolution
            e = torch.transpose(e, 1, 2)
            #print(f"after transpose: {type(e)}, {e.shape}, {e.dtype}, {e.device}")

            # Pass through tanh and max-pooling
            e = self.post_convolution(e)
            #print(f"after tanh and maxpool, before sum: {type(e)}, {e.shape}, {e.dtype}, {e.device}")

            # Sum along columns
            e = e.sum(1)
            #print(f"after sum in channels, before reshape: {type(e)}, {e.shape}, {e.dtype}, {e.device}")

            # Concat all sums together
            e = e.view(cur_batch_size, -1)
            #print(f"after reshape, before concat w/ other filters: {type(e)}, {e.shape}, {e.dtype}, {e.device}")

            # Concatenate with all other filter widths
            outputs = e if outputs == None else torch.cat((outputs, e), 1)
            #print(f"ouputs: {type(outputs)}, {outputs.shape}, {outputs.dtype}, {outputs.device}\n")

        e = outputs
        #print(f"after concat, before highway: {type(e)}, {e.shape}, {e.dtype}, {e.device}")

        # Highway Layers #
        for i in range(0, 2):
            T = self.sigmoid_gate(e)
            C = 1-T
            H = self.tanh_transform(e)
            y = T * H + C * e
            e = y

        #print(f"after highway layer, before final linear: {type(e)}, {e.shape}, {e.dtype}, {e.device}")

        # Final Linear Layer to cast to embed dim
        out = self.linear_layer(e)

        #print(f"after final linear: {type(e)}, {e.shape}, {e.dtype}, {e.device}\n\n\n")

        return out

def pad_collate(batch):
  (x, y) = zip(*batch)

  x_pad = pad_sequence(x, batch_first=True, padding_value=0)
  y_pad = pad_sequence(y, batch_first=True, padding_value=0)

  return x_pad, y_pad


def main(args):
    # Hyper parameters
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

    print("\nStarting training with the following parameters:")
    for k, v in p.items():
        print(str(k) + ': ' + str(v))
    print("\n")

    # Define the dataset, split into train/dev
    print(f"defining dataset...")
    dataset = CustomDataset(args.npy_path)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    print(f'length of train dataset: {len(train_dataset)}, val dataset: {len(val_dataset)}')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p['batch_size'], collate_fn=pad_collate, shuffle=False)
    print(f"num training batches: {len(train_loader)}")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=p['batch_size'], collate_fn=pad_collate, shuffle=False)
    print(f"num validation batches: {len(val_loader)}")

    # Set up model
    model = ConvNet(p).to(device)
    print(f'model is on device: {next(model.parameters()).device}\n')

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])

    # Train model
    stopping_counter = 0
    epoch_train_loss = []
    epoch_dev_loss = []
    for epoch in range(p['epochs']):
        print(f'Starting epoch {epoch}...', flush=True)
        train_loss_list = []
        for i, (word, vec) in enumerate(train_loader): # for each batch

            word = word.to(device)
            vec = vec.to(device)

            # Predict
            outputs = model(word)

            # Get loss and backprop
            vec = torch.reshape(vec, (vec.size(dim=0), p['embed_dim']))
            loss = criterion(outputs, vec)
            train_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'mean train loss for epoch {epoch}: {np.mean(train_loss_list)}', flush=True)
        epoch_train_loss.append(np.mean(train_loss_list))

        # == EVAL STEP ==
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            dev_loss_list = []
            total = 0
            for word, vec in val_loader:
                word = word.to(device)
                vec = vec.to(device)

                outputs = model(word)
                vec = torch.reshape(vec, (vec.size(dim=0), p['embed_dim']))
                loss = criterion(outputs, vec)
                dev_loss_list.append(loss.item())

            print(f'mean dev loss for epoch {epoch}: {np.mean(dev_loss_list)}', flush=True)
            epoch_dev_loss.append(np.mean(dev_loss_list))

        # Automatic stopping logic - this is immensely spaghetti but we go on for now
        if epoch > 10:
            previous = epoch_dev_loss[epoch-11:epoch-1]
            previous_min = min(previous)
            print(f'previous minimum: {previous_min}, length of list: {len(previous)}')

            if epoch_dev_loss[epoch] < previous_min:
                print(f'{epoch_dev_loss[epoch]} is smaller than previous_min {previous_min}\n')
                stopping_counter = 0

                # Save the model checkpoint
                torch.save(model.state_dict(), f'checkpoints/{args.model_name}_best.ckpt')

            else:
                stopping_counter +=1
                print(f'{stopping_counter} epochs without improvement')

            if stopping_counter == 10 and epoch > 25:
                print(f'Have not decreased from best dev loss form 10 epochs and past 25th epoch, terminating...')
                break

        # Put model back into training mode
        model.train()

    # Write epoch/train loss/dev loss to log file
    with open(f"logs/loss/{args.model_name}_training.log", mode="w") as f:
        f.write("epoch,train,dev\n")
        for epoch in range(len(epoch_dev_loss)):
            f.write(f"{epoch},{epoch_train_loss[epoch]},{epoch_dev_loss[epoch]}\n")







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Functionality Options
    #parser.add_argument('--vec_path', dest='vec_path', type=str, help='path to vector file to create map of')
    #parser.add_argument('--charset_path', dest='charset_path', type=str, help='path to charset, will be taken in order')
    parser.add_argument('--npy_path', dest='npy_path', type=str)


    # Training Options
    parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs to (potentially) train')
    parser.add_argument('--auto_stop', action='store_true', help='stop when dev loss doesn\'t improve for 10 epochs')
    parser.add_argument('--filter_interval', dest='filter_interval', type=int, default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)

    # Logging Options
    parser.add_argument('--model_name', dest='model_name', default='default_model', type=str, help='give your model a helpful name!')
    parser.add_argument('--log_dir', dest='log_dir', type=str)
    # Inference
    args = parser.parse_args()
    main(args)