import time
import argparse


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("Timeit - '{}' took {:.4f} seconds".format(method.__name__, te - ts))
        return result
    return timed


def build_arg_parser():
    parser = argparse.ArgumentParser('StancePrediction task')

    parser.add_argument('--rseed', type=int, default=123,
                        help="Random seed ")

    input_group = parser.add_argument_group('Input Data Options')
    input_group.add_argument('--train-file', required=True,
                             help='CSV training input file')
    input_group.add_argument('--test-file', required=True,
                             help='CSV test input file')
    input_group.add_argument('--workdir', default="external/workdir",
                             help='temporary work directory')

    # verbosity options
    verbosity_group = parser.add_argument_group('Verbosity Options')
    verbosity_group.add_argument('-v', "--verbose", action='store_true')
    verbosity_group.add_argument('-d', "--debug", action='store_true',
                                 help="set verbosity to debug")

    return parser
