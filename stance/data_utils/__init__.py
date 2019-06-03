import random
import torch

from itertools import islice


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
