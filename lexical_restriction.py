import os
import sys
from os import listdir
from os.path import isfile,join
from sys import stdout
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix, csc_matrix, vstack
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import argparse

GROUP_ID = "group_id"
FEAT = "feat"


"""
Mean - For each word, the average # of times it occurs.
Usage - % of users (or messages) that used the specific word.
Stdev - For each word, the standard deviation in user-level counts.

"""
SOURCE_MEAN = "Source Mean Frequency"
TARGET_MEAN = "Target Mean Frequency"
SOURCE_MEAN_ZEROREMOVED = "Source Mean Frequency (0-Removed)"
TARGET_MEAN_ZEROREMOVED = "Target Mean Frequency (0-Removed)"
SOURCE_STDEV = "Source Frequency Stdev"
TARGET_STDEV = "Target Frequency Stdev"
SOURCE_STDEV_ZEROREMOVED = "Source Frequency Stdev (0-Removed)"
TARGET_STDEV_ZEROREMOVED = "Target Frequency Stdev (0-Removed)"

SOURCE_USAGE_PERCENT = "Source Usage"
TARGET_USAGE_PERCENT = "Target Usage"



DOMAIN_FREQUENCY_DIFF = "Domain Frequency Diff"
DOMAIN_FREQUENCY_DIFF_ZEROREMOVED = "Domain Frequency Diff (0-Removed)"
DOMAIN_USAGE_LOG_RATIO = "Domain Usage Log Ratio"

PASSES_FREQ_FILTER = "Passes Freq Filter?"
PASSES_FREQ_FILTER_ZEROREMOVED = "Passes Freq Filter? (0-Removed)"
PASSES_USAGE_FILTER = "Passes Usage Filter?"




def fast_variance(sparse_matrix):
    e_x2 = sparse_matrix.multiply(sparse_matrix).mean(axis=0)
    ex = sparse_matrix.mean(axis=0)
    variance = e_x2 - (np.multiply(ex, ex))
    # clip negative variances to 0 - this likely will only
    # happen in cases with precision issues.
    return np.maximum(variance, 0)


def fast_zero_excluded_mean(sparse_matrix):
    M, N = sparse_matrix.shape
    usage_matrix = (sparse_matrix != 0)
    expected_value = sparse_matrix.mean(axis=0)
    nonzero_percent = usage_matrix.mean(axis=0)
    nonzero_percent[nonzero_percent == 0] = 1
    return expected_value / nonzero_percent


def fast_zero_excluded_variance(sparse_matrix):
    M, N = sparse_matrix.shape
    usage_matrix = (sparse_matrix != 0)
    nonzero_percent = usage_matrix.mean(axis=0)
    nonzero_percent[nonzero_percent == 0] = 1
    ex = sparse_matrix.mean(axis=0)
    ex = ex / nonzero_percent
    e_x2 = sparse_matrix.multiply(sparse_matrix).mean(axis=0)
    e_x2 = e_x2/ nonzero_percent
    variance = e_x2 - (np.multiply(ex, ex))
    # clip negative variances to 0 - this likely will only
    # happen in cases with precision issues.
    return np.maximum(variance, 0)


def get_sparse_matrices(source_gram_df, target_gram_df, combined_feat_set=None, count_var="group_norm"):
    """
    Args:
        source_gram_df (pd.DataFrame): Source domain gram_df, same format as a dlatk feature table.
        target_gram_df (pd.DataFrame): Target domain gram_df, same format as a dlatk feature table.
        combined_feat_set (set): The set of all words to include in the analysis (most often, this would be
        the intersection of unique feats in the two dataframes).
        count_var (str): The variable name in the dataframe that is the stand-in for counts.
    Returns:
        (scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, list):
            source, target sparse representations, a consistent list representation of the feature set
            i.e., source and target columns i will both represent the array of all occurences in the
            respective domain of the i'th word in the returned feature_list.
    """
    if combined_feat_set is None or combined_feat_set == "union":
        unique_grams_source = set(source_gram_df[FEAT].unique())
        unique_grams_target = set(target_gram_df[FEAT].unique())
        combined_feat_set = unique_grams_source.union(unique_grams_target)
    elif combined_feat_set == "intersection":
        unique_grams_source = set(source_gram_df[FEAT].unique())
        unique_grams_target = set(target_gram_df[FEAT].unique())
        combined_feat_set = unique_grams_source.intersection(unique_grams_target)
        source_gram_df = source_gram_df[source_gram_df[FEAT].map(
            lambda x: x in combined_feat_set)]
        target_gram_df = target_gram_df[target_gram_df[FEAT].map(
            lambda x: x in combined_feat_set)]

    combined_feat_list = list(sorted(combined_feat_set))
    source_users = source_gram_df[GROUP_ID].unique()
    target_users = target_gram_df[GROUP_ID].unique()

    feat_categories = CategoricalDtype(combined_feat_list, ordered=True)
    source_user_c = CategoricalDtype(sorted(source_users), ordered=True)
    target_user_c = CategoricalDtype(sorted(target_users), ordered=True)

    row_source = source_gram_df[GROUP_ID].astype(source_user_c).cat.codes
    col_source = source_gram_df[FEAT].astype(feat_categories).cat.codes
    row_target = target_gram_df[GROUP_ID].astype(target_user_c).cat.codes
    col_target = target_gram_df[FEAT].astype(feat_categories).cat.codes

    source_sparse_matrix = csc_matrix(
        (source_gram_df[count_var], (row_source, col_source)),
        shape=(source_user_c.categories.size, feat_categories.categories.size))
    target_sparse_matrix = csc_matrix(
        (target_gram_df[count_var], (row_target, col_target)),
        shape=(target_user_c.categories.size, feat_categories.categories.size))
    return source_sparse_matrix, target_sparse_matrix, combined_feat_list


def get_word_statistics(source_matrix,
                        target_matrix,
                        combined_feat_list,
                        stdev_threshold=3,
                        sparse_threshold=10):
    """Given sparse matrices of word frequencies, returns a dataframe with
    means, variances, stdevs, etc. for each word across both domains.

    Args:
        source_matrix (csc_matrix): [description]
        target_matrix (csc_matrix): [description]
        combined_feat_list (list): [description]
    Returns:
        pd.DataFrame: [description]
    """
    # 16to1 representations
    source_usage_matrix = source_matrix != 0
    target_usage_matrix = target_matrix != 0

    source_mean = source_matrix.mean(axis=0)
    target_mean = target_matrix.mean(axis=0)

    source_freq_sigma = np.sqrt(fast_variance(source_matrix))
    target_freq_sigma = np.sqrt(fast_variance(target_matrix))

    source_usage_percent = source_usage_matrix.mean(axis=0)
    target_usage_percent = target_usage_matrix.mean(axis=0)

    source_zeroexcluded_mean = fast_zero_excluded_mean(source_matrix)
    target_zeroexcluded_mean = fast_zero_excluded_mean(target_matrix)

    source_freq_sigma_zeroexcluded = np.sqrt(fast_zero_excluded_variance(source_matrix))
    target_freq_sigma_zeroexcluded = np.sqrt(fast_zero_excluded_variance(target_matrix))

    dataframe = pd.DataFrame(
        np.concatenate((
            source_mean, target_mean,
            source_freq_sigma, target_freq_sigma,
            source_usage_percent, target_usage_percent,
            source_zeroexcluded_mean, target_zeroexcluded_mean,
            source_freq_sigma_zeroexcluded, target_freq_sigma_zeroexcluded
        )).transpose(),
        columns=[SOURCE_MEAN, TARGET_MEAN,
                 SOURCE_STDEV, TARGET_STDEV,
                 SOURCE_USAGE_PERCENT, TARGET_USAGE_PERCENT,
                 SOURCE_MEAN_ZEROREMOVED, TARGET_MEAN_ZEROREMOVED,
                 SOURCE_STDEV_ZEROREMOVED, TARGET_STDEV_ZEROREMOVED
                 ],
        index=combined_feat_list
    )
    # add inferred (score columns):
    dataframe[DOMAIN_USAGE_LOG_RATIO] = np.log10(
        dataframe[TARGET_USAGE_PERCENT]) - np.log10(dataframe[SOURCE_USAGE_PERCENT])

    dataframe[DOMAIN_FREQUENCY_DIFF] = dataframe[TARGET_MEAN] - \
        dataframe[SOURCE_MEAN]
    dataframe[DOMAIN_FREQUENCY_DIFF_ZEROREMOVED] = dataframe[TARGET_MEAN_ZEROREMOVED] - \
        dataframe[SOURCE_MEAN_ZEROREMOVED]

    # add true false columns on whether it passes the filter with default values.
    dataframe[PASSES_FREQ_FILTER] = (
        np.abs(dataframe[DOMAIN_FREQUENCY_DIFF]) < 3*dataframe[SOURCE_STDEV]
    )
    dataframe[PASSES_FREQ_FILTER_ZEROREMOVED] = (
        np.abs(dataframe[DOMAIN_FREQUENCY_DIFF_ZEROREMOVED])
        < stdev_threshold*dataframe[SOURCE_STDEV_ZEROREMOVED]
    )

    dataframe[PASSES_USAGE_FILTER] = (
        np.abs(dataframe[DOMAIN_USAGE_LOG_RATIO]) < np.log10(sparse_threshold)
    )
    dataframe
    return dataframe


def apply_filters(summary_df, sparse_threshold=10, stdev_threshold=3, zero_removed=False) -> pd.DataFrame:
    """Apply a sparse filter and a standard deviation filter.
    Args:
        summary_df ([type]): [description]
        sparse_threshold (float, optional): Keep words where their binary usage between source
        and target are within sparse_threshold multiples of eachother. Defaults to 10.
        stdev_threshold (float, optional): Keep words within stdev_threshold * sigma. Defaults to 3.
        zero_removed (bool, optional). Whether to use 0-removed means and sigmas or regular ones.
    Returns:
        pd.DataFrame: [description]
    """
    source_mean, target_mean, source_sigma = (
        (SOURCE_MEAN_ZEROREMOVED, TARGET_MEAN_ZEROREMOVED, SOURCE_STDEV_ZEROREMOVED) if zero_removed
        else (SOURCE_MEAN, TARGET_MEAN, SOURCE_STDEV)
    )

    return summary_df[
        (
            (summary_df[SOURCE_USAGE_PERCENT] <= sparse_threshold * summary_df[TARGET_USAGE_PERCENT]) &
            (summary_df[TARGET_USAGE_PERCENT] <= sparse_threshold * summary_df[SOURCE_USAGE_PERCENT])
        ) & (
            (summary_df[source_mean] - summary_df[target_mean]).abs() < stdev_threshold*summary_df[source_sigma]
        )
    ]


parser = argparse.ArgumentParser(
    description="Restrict an n-gram feature set by only keeping tokens that occur similarly frequently in both datasets."
)
parser.add_argument("--source_db", required=False, help="Source db in mysql. If specified, a source table is also required.")
parser.add_argument("--target_db", required=False, help="Target db in mysql. If specified, a target table is also required.")
parser.add_argument("--source_table", required=False, help="Source table in mysql. If specified a source db is also required.")
parser.add_argument("--target_table", required=False, help="Target table in mysql. If specified a target db is also required.")
parser.add_argument("--freq_variable", default="group_norm", help="Variable from table to use for frequency.")
parser.add_argument("--sparse_threshold", type=float, default=10, help="Sparse filter threshold. Entries are kept when their binary usage (whether they occur are not) between source and target are within sparse_threshold multiples of eachother.")
parser.add_argument("--stdev_threshold", type=float, default=3, help="Standard Deviation threshold. Keep words that are within {stdev_threshold}-sigma between the source and target domain (with sigma calculated as the stdev on the source domain for that token).")
parser.add_argument("--output", required=False, help="File to output the set of remaining n-grams to. If not specified, will be flushed to stdout.")
parser.add_argument("--combined_feat_set", default="union")
parser.add_argument("--filter_csv", required=False, help="A csv where the FEAT column gives a list of valid tokens")
parser.add_argument("--filter_csv_colname", default="term", help="Column name for filter csv")

parser.add_argument("--zero_removed", default=False, help="For the stdev threshold, determines whether or not to reweight mean and variance to ignore zero entries.")

parser.add_argument("--source_csv", required=False, help="Source CSV, formatted like a dlatk table.")
parser.add_argument("--target_csv", required=False, help="Target CSV, formatted like a dlatk table.")
parser.add_argument("--source_dir", required=False, help="Source CSV directory, with each file formatted like a dlatk table.")
parser.add_argument("--target_dir", required=False, help="Source CSV directory, with each file formatted like a dlatk table.")

parser.add_argument("--output_stats" ,action="store_true", help="Return scores rather than prefiltering.")
parser.add_argument("--debug", action="store_true", help="Enable debug output")

"""
2022-01-26 - Copied from other files for portability reasons.
"""

# USERNAME = os.environ.get("MYSQL_USER")
# PASSWORD = os.environ.get("MYSQL_PASSWORD") or ''
HOST = os.environ.get("MYSQL_HOST") or 'localhost'
MY_CNF = os.environ.get("MYSQL_MY_CNF") or "~/.my.cnf"


def get_engine(database):
    # engine_string = "mysql://{}:{}@{}/{}".format(
    #     USERNAME, PASSWORD, HOST, database
    # )
    db_url = URL(drivername='mysql', host=HOST, database=database,
    query={'read_default_file': MY_CNF})
    return create_engine(name_or_url=db_url)


def sql_to_dataframe(database, table_name, **kwargs):
    engine = get_engine(database)
    dataframe = pd.read_sql(table_name, engine, **kwargs)
    return dataframe


def dataframe_to_sql(database, table_name, df, **kwargs):
    engine = get_engine(database)
    return df.to_sql(table_name, engine, **kwargs)


HEADER = ["group_id", "feat", "value", "group_norm"]

def read_directory(dirpath, header):
    # read every file in a disjoint set of csvs and combine it into a single pandas dataframe
    files = [fname for fname in listdir(dirpath) if isfile(join(dirpath, fname))]
    csv = pd.concat([
       pd.read_csv(join(dirpath, fname), header=None, names=header)
       for fname in files
    ])
    return csv


if __name__ == "__main__":
    args = parser.parse_args()
    source_gram_df = None
    target_gram_df = None
    filter_grams = None

    if (len(sys.argv) < 2):
        parser.print_help()
        exit()


    if args.filter_csv:
        filter_grams = pd.read_csv(args.filter_csv).set_index(args.filter_csv_colname)
        filter_grams = filter_grams.index

    if args.source_csv:
        source_gram_df = pd.read_csv(args.source_csv)
        source_gram_df[FEAT] = source_gram_df[FEAT].astype("str")
        source_gram_df[args.freq_variable] = source_gram_df[args.freq_variable].astype("float32")
    if args.target_csv:
        target_gram_df = pd.read_csv(args.target_csv)
        target_gram_df[FEAT] = target_gram_df[FEAT].astype("str")
        target_gram_df[args.freq_variable] = target_gram_df[args.freq_variable].astype("float32")

    if (args.source_db and args.source_table):
        source_gram_df = sql_to_dataframe(args.source_db, args.source_table)
    if (args.target_db and args.target_table):
        target_gram_df = sql_to_dataframe(args.target_db, args.target_table)

    if args.source_dir:
        source_gram_df = read_directory(args.source_dir, HEADER)
        source_gram_df[FEAT] = source_gram_df[FEAT].astype("str")
        source_gram_df[args.freq_variable] = source_gram_df[args.freq_variable].astype("float32")
    if args.target_dir:
        target_gram_df = read_directory(args.target_dir, HEADER)
        target_gram_df[FEAT] = target_gram_df[FEAT].astype("str")
        target_gram_df[args.freq_variable] = target_gram_df[args.freq_variable].astype("float32")
    if source_gram_df is None or target_gram_df is None:
        print("Must specify input files!")

    if args.filter_csv:
        target_gram_df = target_gram_df[target_gram_df[FEAT].isin(filter_grams)]
        source_gram_df = source_gram_df[source_gram_df[FEAT].isin(filter_grams)]
    source_sparse_matrix, target_sparse_matrix, combined_feat_list = get_sparse_matrices(
        source_gram_df,
        target_gram_df,
        combined_feat_set=args.combined_feat_set,
        count_var=args.freq_variable
    )
    statistics = get_word_statistics(
        source_sparse_matrix,
        target_sparse_matrix,
        combined_feat_list,
        args.stdev_threshold,
        args.sparse_threshold,
    )
    if args.debug:
        print(statistics)
    if args.output_stats:
        # TODO - name the feat column / flatten it out of index
        statistics.to_csv(args.output)
    else:
        filtered_words = apply_filters(
            statistics,
            args.sparse_threshold,
            args.stdev_threshold,
            args.zero_removed
        )
        if args.debug:
            print(filtered_words)
        output = filtered_words[[]].reset_index()["index"]
        if args.output is None:
            output.to_csv(stdout, header=False, index=False)
        else:
            output.to_csv(args.output, header=False, index=False)
