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

def plot_2B(df2_charCNN, df2_SAM):
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(data=df2_charCNN, x='Model', y='Cosine Sim', ax=ax[0])
    sns.lineplot(data=df2_charCNN, x='Model', y='P@10', ax=ax[0])
    sns.lineplot(data=df2_charCNN, x='Model', y='P@100', ax=ax[0])
    sns.lineplot(data=df2_SAM, x='Model', y='Cosine Sim', ax=ax[1])
    sns.lineplot(data=df2_SAM, x='Model', y='P@10', ax=ax[1])
    sns.lineplot(data=df2_SAM, x='Model', y='P@100', ax=ax[1])

    ax[0].set_yticks(np.arange(0, 1, step=0.1))
    ax[1].set_yticks(np.arange(0, 1, step=0.1))
    ax[0].set_xticklabels(labels=['charCNN', '+1', '+2', '+3', '+4'])
    ax[1].set_xticklabels(labels=['SAM', '+1', '+2', '+3', '+4'])

    plt.show()

def plot_2C(df2_charCNN, df2_SAM):
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(data=df2_charCNN, x='Model', y='test ppl (f)', ax=ax[0])
    sns.lineplot(data=df2_charCNN, x='Model', y='valid ppl (f)', ax=ax[0])
    sns.lineplot(data=df2_SAM, x='Model', y='test ppl (f)', ax=ax[1])
    sns.lineplot(data=df2_SAM, x='Model', y='valid ppl (f)', ax=ax[1])

    ax[0].set_yticks(np.arange(120, 200, step=20))
    ax[1].set_yticks(np.arange(120, 200, step=20))
    ax[0].set_xticklabels(labels=['charCNN', '+1', '+2', '+3', '+4'])
    ax[1].set_xticklabels(labels=['SAM', '+1', '+2', '+3', '+4'])

    plt.show()

def plot_3A(df3_SAM):
    # fig, ax = plt.subplots(1, 2)
    # sns.lineplot(data=df3_SAM, x='Model', y='Cosine Sim', ax=ax[0])

    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data = df3_SAM, x='Model', y='Size', ax=ax)

    ax2 = ax.twinx()
    sns.lineplot(data=df3_SAM, x='Model', y='Throughput', ax=ax2)

    plt.show()

    pass

def plot_3B(df3_SAM3, df3_SAM4):
    fig, ax = plt.subplots(2, 2)

    sns.lineplot(data = df3_SAM3, x = 'Model', y='Cosine Sim', ax=ax[0, 0])
    sns.lineplot(data = df3_SAM3, x = 'Model', y='P@10', ax=ax[0, 0])
    sns.lineplot(data = df3_SAM3, x = 'Model', y='P@100', ax=ax[0, 0])
    sns.lineplot(data=df3_SAM4, x='Model', y='Cosine Sim', ax=ax[0, 1])
    sns.lineplot(data=df3_SAM4, x='Model', y='P@10', ax=ax[0, 1])
    sns.lineplot(data=df3_SAM4, x='Model', y='P@100', ax=ax[0, 1])

    sns.lineplot(data=df3_SAM3, x='Model', y='test ppl (f)', ax=ax[1, 0])
    sns.lineplot(data=df3_SAM3, x='Model', y='valid ppl (f)', ax=ax[1, 0])
    sns.lineplot(data=df3_SAM4, x='Model', y='test ppl (f)', ax=ax[1, 1])
    sns.lineplot(data=df3_SAM4, x='Model', y='valid ppl (f)', ax=ax[1, 1])

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


    # Experiment 1 Plots


    # Experiment 2 Plots
    # For all, put SAM on left and Char CNN on right
    # 2A - Size on Left
    #plot_2A(df2_charCNN, df2_SAM)

    # 2B - CosSim, P@10, P@100 on Left
    #plot_2B(df2_charCNN, df2_SAM)

    # 2C - Frozen PPL on Left
    #plot_2C(df2_charCNN, df2_SAM)

    # Experiment 3 Plots

    # Report Size and Throughput of +0
    #plot_3A(df3_SAM)

    # Report Recon Integrity and Perplexity
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


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# def graph_grid(csv_path):
#
#     table = pd.read_csv(csv_path)
#     table['WER'] = table['WER']*100
#
#     #table.columns = [c.replace(' ', '_') for c in table.columns]
#     grouped_final = table.groupby(table.ngram)
#     no_gram = grouped_final.get_group(0)
#     unigram = grouped_final.get_group(1)
#
#     # fig, ax = plt.subplots()
#     # for final in ['False', 'True']:
#     #     ax.plot(no_gram[no_gram.final==final]['self-loop'], no_gram[no_gram.final==final].WER,label=final)
#     fig, ax = plt.subplots(1, 2)
#     fig.subplots_adjust(wspace=0.0125)
#     sns.set_style("ticks")
#     sns.lineplot(data=no_gram, x='self-loop', y='WER', style='final', hue='final', errorbar=None, ax=ax[0], markers=True)
#     sns.lineplot(data=unigram, x='self-loop', y='WER', style='final', hue='final', errorbar=None, ax=ax[1], markers=True)
#
#     ax[0].title.set_text("using uniform unigram distribution")
#     ax[1].title.set_text("using unigram LM")
#
#     ax[0].legend(title = 'using final probability')
#     ax[1].legend(title = 'using final probability')
#
#     ax[0].set(xlabel="self-loop probability")
#     ax[1].set(xlabel="self-loop probability")
#
#     ax[0].set_yticks(range(50, 300, 50), labels=range(50, 300, 50))
#     ax[1].set_yticks(range(50, 300, 50), labels=range(50, 300, 50))
#
#     ax[1].set(yticklabels=[])
#     ax[1].set(ylabel=None)
#     ax[1].tick_params(left=False)
#
#     plt.show()
#
# def graph_beam(csv_path):
#     table = pd.read_csv(csv_path)
#     table['WER'] = table['WER'] * 100
#     grouped_beam = table.groupby(table.beam_type)
#     mbeam = grouped_beam.get_group('mbeam')
#     bbeam = grouped_beam.get_group('bbeam')
#
#
#     # fig, ax = plt.subplots()
#     # for final in ['False', 'True']:
#     #     ax.plot(no_gram[no_gram.final==final]['self-loop'], no_gram[no_gram.final==final].WER,label=final)
#     fig, ax = plt.subplots(2, 2)
#     fig.subplots_adjust(hspace = 0.0925, wspace=0.0625)
#
#
#     sns.lineplot(data=mbeam, x='beam', y='WER', hue='pruning_interval', style='pruning_interval', markers=True, errorbar=None, ax=ax[0,0])
#     sns.lineplot(data=bbeam, x='beam', y='WER', hue='pruning_interval', style='pruning_interval', markers=True, errorbar=None, ax=ax[0,1])
#     sns.lineplot(data=mbeam, x='beam', y='forward', hue='pruning_interval', style='pruning_interval', markers=True, errorbar=None, ax=ax[1,0])
#     sns.lineplot(data=bbeam, x='beam', y='forward', hue='pruning_interval', style='pruning_interval', markers=True,  errorbar=None, ax=ax[1,1])
#
#     ax[0, 0].axhline(y=41.99, linestyle='--', color='deepskyblue', alpha=0.75)
#     ax[0, 1].axhline(y=41.99, linestyle='--', color='deepskyblue', alpha=0.75)
#     ax[1, 0].axhline(y=173370, linestyle='--', color='deepskyblue', alpha=0.75)
#     ax[1, 1].axhline(y=173370, linestyle='--', color='deepskyblue', alpha=0.75)
#
#     ax[0, 0].set_yticks(range(30, 100, 10), labels=range(30, 100, 10))
#     ax[0, 1].set_yticks(range(30, 100, 10), labels=range(30, 100, 10))
#     ax[1, 0].set_yticks(range(0, 200000, 20000), labels=range(0, 200000, 20000))
#
#     # This is kinda misleading but still accurate
#     ax[1, 1].set_yticks(range(0, 200000, 20000), labels=range(0, 200000, 20000))
#
#     #ax[0, 0].text(3 + 0.2, 4.5, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')
#     # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     # ax[0, 0].text(0.05, 70,  transform=ax[0,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
#     ax[0, 0].annotate('benchmark', xy=(0, 36), xytext=(0, 36), color='deepskyblue', size="medium")
#
#     ax[0, 0].title.set_text("mbeam")
#     ax[0, 1].title.set_text("bbeam")
#     ax[1, 0].set(xlabel='beam width')
#     ax[1, 1].set(xlabel='beam width')
#     ax[1, 0].set(ylabel='forward computations')
#
#     ax[0, 0].set(xticklabels=[])
#     ax[0, 0].set(xlabel=None)
#     ax[0, 0].tick_params(bottom=False)
#     ax[0, 0].get_legend().remove()
#
#     ax[0, 1].legend(title='pruning interval')
#     ax[0, 1].set(xticklabels=[])
#     ax[0, 1].set(xlabel=None)
#     ax[0, 1].tick_params(bottom=False)
#     ax[0, 1].set(yticklabels=[])
#     ax[0, 1].set(ylabel=None)
#     ax[0, 1].tick_params(left=False)
#
#     ax[1, 0].get_legend().remove()
#
#     ax[1, 1].set(yticklabels=[])
#     ax[1, 1].set(ylabel=None)
#     ax[1, 1].tick_params(left=False)
#     ax[1, 1].get_legend().remove()
#
#     plt.show()
#
# if __name__ == "__main__":
#     # graph_grid('results/csv/trimmed_grid_30.csv')
#     # graph_beam('results/csv/beam_30.csv')
#
#     #graph_grid('results/csv/grid_160.csv')
#     graph_beam('results/csv/beam_combined_160.csv')

