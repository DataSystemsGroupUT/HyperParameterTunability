import Orange
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman


def do_friedman_test(ranked_data):
    return friedmanchisquare(
        *[group for idx, group in ranked_data.iloc[:, 1:].iteritems()]
    )


def do_nemenyi_test(ranked_data, plot=False):
    ranks_per_dataset = ranked_data.iloc[:, 1:]

    if plot:
        names = list(ranked_data.columns)[1:]
        avg_ranks = ranks_per_dataset.mean(axis=0)
        cd = Orange.evaluation.compute_CD(
            avg_ranks, ranked_data.shape[0], alpha='0.05', test='nemenyi')
        Orange.evaluation.graph_ranks(
            avg_ranks, names, cd=cd, width=10, textspace=1.5)

        plt.show()

    return posthoc_nemenyi_friedman(ranks_per_dataset)


def plot_heatmap(rank_frequences, save_plot=False):
    ax = sns.heatmap(rank_frequences, cmap=sns.cm.rocket_r,
                     annot=True, fmt='g')
    plt.title('Heat Map for Algorithm Performance Analysis')
    plt.xlabel('Algorithm Name')
    plt.ylabel('Rank')
    plt.tight_layout()
    if save_plot:
        plt.savefig('./output_plots/heatmap_dec2.png', dpi=fig.dpi)
    plt.show()
