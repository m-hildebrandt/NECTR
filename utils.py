import numpy as np
import pandas as pd
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Utility/Helper methods


def parse_exponential(x):
    """
    Fetches the whole and exponent of x. E.g. If x=1e-5, returns (1,-5)
    :param x: Number in scientific notation
    :return: Tuple consisting of whole and exponent part
    """
    exp = np.floor(np.log10(np.abs(x)))
    return (x / (10 ** exp)).astype(int), exp.astype(int)


def generate_range(a, b, dtype='int', step=None):
    """
    Returns a range of numbers between a(as minimum) and b(as maximum)
    :param a: Minimum of the range
    :param b: Maximum of the range
    :param dtype: Data-types of a,b and of the range
    :param step: If specified, it is equivalent to range(a, b, step).
    Else, step will be randomly generated based on a and b
    :return: List consisting of the range of numbers between a and b
    """
    parameters = []
    if dtype == 'int':
        if step is None:
            high = -((a - b) // 2)
            if high == 1:
                high += 1
            elif high == 0:
                high += 2
            step = np.random.randint(1, high)
        parameters = list(range(a, b, step))
        parameters.append(b)
    elif dtype == 'exp':
        a_whole, a_exp = parse_exponential(a)
        b_whole, b_exp = parse_exponential(b)
        temp = itertools.product(generate_range(a_whole, b_whole), generate_range(a_exp, b_exp))
        parameters = [i ** float(j) if i != 1 else i * 10 ** float(j) for i, j in temp]
    return parameters


def extract_metrics(results):
    """
    Extract the performance metrics from the results
    :param results: Results obtained after cross validation
    :return: all_metrics: Metrics for all models evaluated as part of cross validation
    :return: best_metrics: Metrics of the model chosen by cross validation as the best model
    """
    all_metrics = None

    names = ["mean_rank_raw", "median_rank_raw", "mrr_raw", "hits_top10_raw", "hits_top10_perc_raw",
             "histogram_perc_rank_raw",
             "histogram_num_rank_raw",
             "mean_rank_in_category_raw", "median_rank_in_category_raw", "mrr_in_category_raw",
             "hits_top10_in_category_raw", "hits_top10_perc_in_category_raw",
             "histogram_perc_rank_in_category_raw", "histogram_num_rank_in_category_raw", "mean_rank_filtered",
             "median_rank_filtered", "mrr_filtered", "hits_top10_filtered", "hits_top10_perc_filtered",
             "histogram_perc_rank_filtered",
             "histogram_num_rank_filtered", "mean_rank_in_category_filtered", "median_rank_in_category_filtered",
             "mrr_in_category_filtered", "hits_top10_in_category_filtered", "hits_top10_perc_in_category_filtered",
             "histogram_perc_rank_in_category_filtered",
             "histogram_num_rank_in_category_filtered", "recall_raw", "recall_filtered"]

    histograms = ["histogram_perc_rank_raw", "histogram_num_rank_raw", "histogram_perc_rank_in_category_raw",
                  "histogram_num_rank_in_category_raw", "histogram_perc_rank_filtered", "histogram_num_rank_filtered",
                  "histogram_perc_rank_in_category_filtered", "histogram_num_rank_in_category_filtered"]

    if results['validation']:
        metrics = [[i['parameter'], i['metrics']] for i in results['validation']]

        df_metrics = pd.DataFrame(metrics, columns=['parameter', 'metrics'])
        df_metrics.index = df_metrics.parameter
        df_metrics.drop('parameter', axis=1, inplace=True)

        temp = df_metrics['metrics'].apply(lambda x: pd.Series(x)).rename(
            columns={i: names[i] for i in range(len(names))})

        df_metrics = pd.concat([df_metrics, temp], axis=1).drop('metrics', axis=1)

        df_histograms = df_metrics[histograms]
        df_metrics = df_metrics[df_metrics.columns.difference(histograms)].T

        all_metrics = df_metrics

    metrics = [[results['best'], results['test']]]

    df_metrics = pd.DataFrame(metrics, columns=['parameter', 'metrics'])
    df_metrics.index = df_metrics.parameter
    df_metrics.drop('parameter', axis=1, inplace=True)

    temp = df_metrics['metrics'].apply(lambda x: pd.Series(x)).rename(columns={i: names[i] for i in range(len(names))})

    df_metrics = pd.concat([df_metrics, temp], axis=1).drop('metrics', axis=1)

    df_histograms = df_metrics[histograms]
    df_metrics = df_metrics[df_metrics.columns.difference(histograms)].T

    best_metrics = df_metrics

    return all_metrics, best_metrics


def visualize_item_embeddings(item_embeddings, categories, path_to_results=''):
    """
    Plots the item embeddings as a scatter plot colored by category
    :param item_embeddings: Item embeddings in >2 dimensions
    :param categories: Item categories used to color the items
    :param path_to_results: Path to save the image of the scatter plot
    :return: None
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(item_embeddings)

    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]
    unique = list(set(categories))
    colors = [plt.cm.jet(float(i + 1) / len(unique)) for i, _ in enumerate(unique)]
    for i, u in enumerate(unique):
        xi = [x[j] for j in range(len(x)) if categories[j] == u]
        yi = [y[j] for j in range(len(x)) if categories[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(u))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Items by Category', fontweight='bold')
    if path_to_results:
        plt.savefig(path_to_results + '_items_by_category.png', bbox_inches='tight')
    else:
        plt.show()
    plt.close()
