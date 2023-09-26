import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse, os

def plot_2A(df2_charCNN, df2_SAM):
    fig, ax = plt.subplots(1, 2)
    #fig.subplots_adjust(wspace=0.0125)
    #sns.set_style("ticks")
    #sns.lineplot(data=no_gram, x='self-loop', y='WER', style='final', hue='final', errorbar=None, ax=ax[0], markers=True)
    sns.lineplot(data=df2_charCNN, x='Model', y='Size', ax=ax[0])
    sns.lineplot(data=df2_SAM, x='Model', y='Size', ax=ax[1])

    plt.savefig("plots/plot_2A.svg")
    plt.show()
    pass

def plot_2A(df2_charCNN, df2_SAM):
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(data=df2_charCNN, x='Model', y='Cosine Sim', color='purple', marker='.', markersize=20, ax=ax[0])
    sns.lineplot(data=df2_charCNN, x='Model', y='P@100', color='red', marker='.', markersize=20, ax=ax[0])
    sns.lineplot(data=df2_charCNN, x='Model', y='P@10', color='darkorange', marker='.', markersize=20, ax=ax[0])

    sns.lineplot(data=df2_SAM, x='Model', y='Cosine Sim', color='purple', marker='.', markersize=20, ax=ax[1])
    sns.lineplot(data=df2_SAM, x='Model', y='P@100', color='red', marker='.', markersize=20, ax=ax[1])
    sns.lineplot(data=df2_SAM, x='Model', y='P@10', color='darkorange', marker='.', markersize=20, ax=ax[1])

    ax[0].set_yticks(np.arange(0, 1.1, step=0.1))
    ax[1].set_yticks(np.arange(0, 1.1, step=0.1))
    ax[0].set_xticklabels(labels=['Base', '+1', '+2', '+3', '+4'])
    ax[1].set_xticklabels(labels=['Base', '+1', '+2', '+3', '+4'])



    # Plot sizes on right axis
    ax2 = ax[0].twinx()
    sns.lineplot(data=df2_charCNN, x='Model', y='Size', color='deepskyblue', marker='o', markersize=20, linewidth=5, alpha=0.25, ax=ax2)

    ax3 = ax[1].twinx()
    sns.lineplot(data=df2_SAM, x='Model', y='Size', color='deepskyblue', marker='o', markersize=20, linewidth=5, alpha=0.25, ax=ax3)

    ax2.set_yticks(np.arange(0, 1100, 100))
    ax3.set_yticks(np.arange(0, 1100, 100))

    # Add size benchmarks
    ax2.axhline(y=931, linestyle='--', linewidth=5, color='deepskyblue',  alpha=0.25)
    ax3.axhline(y=931, linestyle='--', linewidth=5, color='deepskyblue',  alpha=0.25)

    # Clean up and position plots
    ax[1].set(yticklabels=[])
    ax[1].set(ylabel=None)
    ax[1].tick_params(left=False)

    ax[0].title.set_text("charCNN")
    ax[0].set(xlabel='Hybrid Model Variation')
    ax[1].title.set_text("SAM")
    ax[1].set(xlabel='Hybrid Model Variation')

    ax[0].set(ylabel='')
    ax3.set(ylabel='Size (MB)')

    ax2.set(yticklabels=[])
    ax2.set(ylabel=None)
    ax2.tick_params(right=False)

    fig.subplots_adjust(wspace=0.0125)

    plt.show()

def plot_2B(df2_charCNN, df2_SAM):
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(data=df2_charCNN, x='Model', y='test ppl (f)', color='forestgreen', marker='.', markersize=20, ax=ax[0])
    sns.lineplot(data=df2_charCNN, x='Model', y='valid ppl (f)', color='saddlebrown', marker='.', markersize=20, ax=ax[0])
    sns.lineplot(data=df2_SAM, x='Model', y='test ppl (f)', color='forestgreen', marker='.', markersize=20, ax=ax[1])
    sns.lineplot(data=df2_SAM, x='Model', y='valid ppl (f)', color='saddlebrown', marker='.', markersize=20, ax=ax[1])

    ax[0].set_yticks(np.arange(120, 200, step=10))
    ax[1].set_yticks(np.arange(120, 200, step=10))
    ax[0].set_xticklabels(labels=['Base', '+1', '+2', '+3', '+4'])
    ax[1].set_xticklabels(labels=['Base', '+1', '+2', '+3', '+4'])

    # Add perplexity benchmarks
    ax[0].axhline(y=137, linestyle='--', color='forestgreen', alpha=1)
    ax[0].axhline(y=127, linestyle='--', color='saddlebrown', alpha=1)
    ax[1].axhline(y=137, linestyle='--', color='forestgreen', alpha=1)
    ax[1].axhline(y=127, linestyle='--', color='saddlebrown', alpha=1)

    # Plot sizes on right axis
    ax2 = ax[0].twinx()
    sns.lineplot(data=df2_charCNN, x='Model', y='Size', color='deepskyblue', marker='o', markersize=20, linewidth=5, alpha=0.25, ax=ax2)

    ax3 = ax[1].twinx()
    sns.lineplot(data=df2_SAM, x='Model', y='Size', color='deepskyblue', marker='o', markersize=20, linewidth=5, alpha=0.25, ax=ax3)

    ax2.set_yticks(np.arange(0, 1100, 100))
    ax3.set_yticks(np.arange(0, 1100, 100))

    # Add size benchmarks
    ax2.axhline(y=931, linestyle='--', linewidth=5, color='deepskyblue', alpha=0.25)
    ax3.axhline(y=931, linestyle='--', linewidth=5, color='deepskyblue', alpha=0.25)

    # Clean up and position plots
    ax[1].set(yticklabels=[])
    ax[1].set(ylabel=None)
    ax[1].tick_params(left=False)

    ax[0].title.set_text("charCNN")
    ax[0].set(xlabel='Hybrid Model Variation')
    ax[1].title.set_text("SAM")
    ax[1].set(xlabel='Hybrid Model Variation')

    ax[0].set(ylabel='')
    ax3.set(ylabel='Size (MB)')

    ax2.set(yticklabels=[])
    ax2.set(ylabel=None)
    ax2.tick_params(right=False)

    fig.subplots_adjust(wspace=0.0125)

    plt.show()

def plot_3A(df3_SAM):
    # fig, ax = plt.subplots(1, 2)
    # sns.lineplot(data=df3_SAM, x='Model', y='Cosine Sim', ax=ax[0])

    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data = df3_SAM, x='Model', y='Throughput', linewidth=5, marker='o', markersize=10, color='slategrey', ax=ax)
    ax.set_yticks(np.arange(45, 75, 5))
    ax.axhline(y=67.7, linestyle='--', linewidth=2.5, color='slategrey', alpha=1)
    ax.set(ylabel='Throughput (k words/sec)')

    ax2 = ax.twinx()
    sns.lineplot(data=df3_SAM, x='Model', y='Size', linewidth=5, marker='o', markersize=10, color='deepskyblue', ax=ax2)
    ax2.set_yticks(np.arange(0, 1100, 100))
    ax2.axhline(y=931, linestyle='--', linewidth=2.5, color='deepskyblue', alpha=1)
    ax2.set(ylabel='Size (MB)')

    ax.title.set_text("Size and Throughput of SAM")
    ax.set_xticklabels(labels=['200k', '100k', '10k', '1k', '100'])
    ax.set(xlabel='Number of Shared Embeddings (H)')

    plt.show()

    pass

def plot_3B(df3_SAM3, df3_SAM4):
    fig, ax = plt.subplots(2, 2)

    sns.lineplot(data=df3_SAM3, x='Model', y='Cosine Sim', marker='.', markersize=10, color='purple', ax=ax[0, 0])
    sns.lineplot(data=df3_SAM3, x='Model', y='P@100', marker='.', markersize=10, color='red', ax=ax[0, 0])
    sns.lineplot(data=df3_SAM3, x='Model', y='P@10', marker='.', markersize=10, color='darkorange', ax=ax[0, 0])

    sns.lineplot(data=df3_SAM4, x='Model', y='Cosine Sim', marker='.', markersize=10, color='purple', ax=ax[0, 1])
    sns.lineplot(data=df3_SAM4, x='Model', y='P@100', marker='.', markersize=10, color='red', ax=ax[0, 1])
    sns.lineplot(data=df3_SAM4, x='Model', y='P@10', marker='.', markersize=10, color='darkorange', ax=ax[0, 1])


    sns.lineplot(data=df3_SAM3, x='Model', y='test ppl (f)', marker='.', markersize=20, color='forestgreen', ax=ax[1, 0])
    sns.lineplot(data=df3_SAM3, x='Model', y='valid ppl (f)', marker='.', markersize=20, color='saddlebrown', ax=ax[1, 0])
    sns.lineplot(data=df3_SAM4, x='Model', y='test ppl (f)', marker='.', markersize=20, color='forestgreen', ax=ax[1, 1])
    sns.lineplot(data=df3_SAM4, x='Model', y='valid ppl (f)', marker='.', markersize=20, color='saddlebrown', ax=ax[1, 1])

    ax[1, 0].axhline(y=137, linestyle='--', color='forestgreen', alpha=1)
    ax[1, 0].axhline(y=127, linestyle='--', color='saddlebrown', alpha=1)
    ax[1, 1].axhline(y=137, linestyle='--', color='forestgreen', alpha=1)
    ax[1, 1].axhline(y=127, linestyle='--', color='saddlebrown', alpha=1)

    # Right Axis of [0,0]
    ax00 = ax[0, 0].twinx()
    sns.lineplot(data=df3_SAM3, x='Model', y='Size', linewidth=5, marker='o', markersize=10, color='deepskyblue', alpha=0.25, ax=ax00)
    ax00.set_yticks(np.arange(0, 1100, 200))
    ax00.axhline(y=931, linestyle='--', linewidth=2.5, color='deepskyblue', alpha=0.25)
    ax00.set(ylabel='Size (MB)')

    # Right Axis of [0,1]
    ax01 = ax[0, 1].twinx()
    sns.lineplot(data=df3_SAM4, x='Model', y='Size', linewidth=5, marker='o', markersize=10, color='deepskyblue', alpha=0.25, ax=ax01)
    ax01.set_yticks(np.arange(0, 1100, 200))
    ax01.axhline(y=931, linestyle='--', linewidth=2.5, color='deepskyblue', alpha=0.25)

    # Right Axis of [1,0]
    ax10 = ax[1, 0].twinx()
    sns.lineplot(data=df3_SAM3, x='Model', y='Size', linewidth=5, marker='o', markersize=10, color='deepskyblue', alpha=0.25, ax=ax10)
    ax10.set_yticks(np.arange(0, 1100, 200))
    ax10.axhline(y=931, linestyle='--', linewidth=2.5, color='deepskyblue', alpha=0.25)

    # Right Axis of [1,1]
    ax11 = ax[1, 1].twinx()
    sns.lineplot(data=df3_SAM4, x='Model', y='Size', linewidth=5, marker='o', markersize=10, color='deepskyblue', alpha=0.25, ax=ax11)
    ax11.set_yticks(np.arange(0, 1100, 200))
    ax11.axhline(y=931, linestyle='--', linewidth=2.5, color='deepskyblue', alpha=0.25)

    # Clean up of axes
    ax[0, 0].set_yticks(np.arange(0, 1.1, step=0.2))
    ax[0, 1].set_yticks(np.arange(0, 1.1, step=0.2))
    ax[1, 0].set_yticks(np.arange(120, 210, step=10))
    ax[1, 1].set_yticks(np.arange(120, 210, step=10))

    ax00.set(yticklabels=[])
    ax00.set(ylabel=None)
    ax00.tick_params(right=False)
    ax[0, 0].set(xticklabels=[])
    ax[0, 0].set(xlabel=None)
    ax[0, 0].tick_params(bottom=False)

    ax[0, 1].set(yticklabels=[])
    ax[0, 1].set(ylabel=None)
    ax[0, 1].tick_params(left=False)
    ax[0, 1].set(xticklabels=[])
    ax[0, 1].set(xlabel=None)
    ax[0, 1].tick_params(bottom=False)

    ax[1, 1].set(yticklabels=[])
    ax[1, 1].set(ylabel=None)
    ax[1, 1].tick_params(left=False)

    ax[0, 0].title.set_text("SAM+3")
    ax[1, 0].set(xlabel='Number of Shared Embeddings (H)')
    ax[0, 1].title.set_text("SAM+4")
    ax[1, 1].set(xlabel='Number of Shared Embeddings (H)')

    ax10.set(yticklabels=[])
    ax10.set(ylabel=None)
    ax10.tick_params(right=False)

    ax[0, 0].set(ylabel='')
    ax[1, 0].set(ylabel='')
    ax01.set(ylabel='Size (MB)')
    ax11.set(ylabel='Size (MB)')

    ax[0, 0].set_xticklabels(labels=['200k', '100k', '10k', '1k', '100'])
    ax[0, 0].set_yticklabels(labels=['', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax[0, 1].set_xticklabels(labels=['200k', '100k', '10k', '1k', '100'])
    ax01.set_yticklabels(labels=['', '200', '400', '600', '800', '1000'])

    ax[1, 0].set_xticklabels(labels=['200k', '100k', '10k', '1k', '100'])
    ax[1, 1].set_xticklabels(labels=['200k', '100k', '10k', '1k', '100'])

    fig.subplots_adjust(wspace=0.0125)
    fig.subplots_adjust(hspace=0.0250)

    plt.show()

def main(args):
    # Read in data and put in csv
    df = pd.read_csv('data/results.csv').dropna()

    # Preprocessing
    df['Size'] = df['Size'].str.replace('\(.*\)', '', regex=True).replace('[MB<\s]', '', regex=True).astype('int')
    df['Throughput'] = df['Throughput'].str.replace('[\s<kKw\/s]', '', regex=True).astype('double')

    # Experiment 1
    df1 = df.iloc[:4, :]

    # Experiment 2
    df2_SAM = df.iloc[4:9, :]
    df2_charCNN = df.iloc[9:14, :]

    # Experiment 3
    df3_SAM = df.iloc[14:19, :]
    df3_SAM3 = df.iloc[19:24, :]
    df3_SAM4 = df.iloc[24:29, :]

    # Experiment 2 Plots
    #plot_2A(df2_charCNN, df2_SAM)
    #plot_2B(df2_charCNN, df2_SAM)

    # Experiment 3 Plots
    #plot_3A(df3_SAM)
    plot_3B(df3_SAM3, df3_SAM4)

    return None


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # # Functionality Options
    # parser.add_argument('--vec_path', dest='vec_path', type=str, default='', help='path of original vectors')
    # parser.add_argument('--write_path', dest='write_path', type=str)
    #
    # # Pass args to main
    # args = parser.parse_args()
    args = None
    main(args)

