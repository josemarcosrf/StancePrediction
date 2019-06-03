import csv
import random
import torch
import numpy as np
import pandas as pd

from itertools import islice


def load_from_csv(data_file):
    """Reads from a CSV file and loads the dataset into a pandas dataframe

    Returns:
        pandas.dataframe: dataframe containing the Stance Dataset
    """
    data = []
    with open(data_file, 'r',  encoding="iso-8859-1") as fin:
        reader = csv.reader(fin, quotechar='"')
        columns = next(reader)
        for line in reader:
            data.append(line)

    data_df = pd.DataFrame(data, columns=columns)

    print("Read a total of {} data points".format(len(data_df)))
    data_df.head()

    return data_df


def load_from_txt(data_file):
    """Reads from a TXT file and loads the dataset into a pandas dataframe

    Args:
        data_file ([type]): [description]
    """
    data = []
    with open(data_file, 'r',  encoding="iso-8859-1") as fin:
        reader = csv.reader(fin, delimiter='\t')
        columns = next(reader)
        for line in reader:
            data.append(line)

    data_df = pd.DataFrame(data, columns=columns)

    print("Read a total of {} data points".format(len(data_df)))
    data_df.head()

    return data_df


def chunk(iterable, c_size, stack_func=None):
    """
    Given an iterable yields chunks of size 'c_size'.
    The iterable can be an interator, we do not assume iterable to have
    'len' method.
    Args:
        iterable (iterable): to be partitioned in chunks
        c_size (int): size of the chunks to be produced
    Returns:
        (generator) of elements of size 'c_size' from the given iterable
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, c_size))
        if not chunk:
            return
        if stack_func:
            yield stack_func(chunk)
        else:
            yield chunk


def parallel_shuffle(*args):
    """
    Shuffle n lists concurrently.

    Args:
        *args: list of iterables to shuffle concurrently

    Returns:
        shuffled iterables
    """
    combined = list(zip(*args))
    random.shuffle(combined)
    args = zip(*combined)
    return [list(x) for x in args]


def parallel_split(split_ratio, *args):
    """
    Splits n lists concurrently

    Args:
        *args: list of iterables to split
        split_ratio (float): proportion to split the lists into two parts
    Returns:

    """
    all_outputs = []
    for a in args:
        split_idx = int(len(a) * split_ratio)
        all_outputs.append((a[:split_idx], a[split_idx:]))
    return all_outputs


def to_tensor(ndarray):
    """Converts a np.array into pytorch.tensor

    Args:
        ndarray (np.array): numpy array to convert to tensor
    """
    return torch.from_numpy(ndarray)
