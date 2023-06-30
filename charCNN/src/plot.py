import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def main(args):
    df = pd.read_csv(args.log_path)
    print(df)
    plt.plot(df['epoch'], df['dev'])
    plt.plot(df['epoch'], df['train'])
    plt.legend(['devloss','trainloss'])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Functionality Options
    parser.add_argument('--log_path', dest='log_path', type=str, help='log path in the format epoch,trainloss,devloss\\n')

    # Inference
    args = parser.parse_args()
    main(args)