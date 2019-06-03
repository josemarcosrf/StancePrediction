import os
import argparse
import coloredlogs
import logging

import numpy as np

from stance import timeit
from stance.encoders import LaserEncoder
from stance.data_utils.preprocessor import Preprocessor
from stance.data_utils.stance_batcher import StanceBatcher


logger = logging.getLogger(__name__)


@timeit
def prepare_save_or_load(batcher):

    # TODO: Refactor all this mess!

    encoded_train_inputs_path = os.path.join(args.workdir,
                                             "laser_training_inputs.npy")
    encoded_test_inputs_path = os.path.join(args.workdir,
                                            "laser_test_inputs.npy")

    if not os.path.exists(encoded_train_inputs_path) or \
            not os.path.exists(encoded_test_inputs_path):

        # Preprocess and encode the inputs (tweets)
        batcher.prepare_inputs()

        # Save to avoid re-encoding
        if not os.path.exists(encoded_train_inputs_path):
            np.save(encoded_train_inputs_path, batcher.train_data_encoded)
        if not os.path.exists(encoded_test_inputs_path):
            np.save(encoded_test_inputs_path, batcher.test_data_encoded)

    else:
        batcher.load_encoded_inputs(encoded_train_inputs_path,
                                    encoded_test_inputs_path)

    batcher.prepare_outputs()


@timeit
def make_classifier_and_classify(batcher, clf_name="mlp"):
    from sklearn.metrics import classification_report

    if clf_name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=100, n_estimators=1000,
                                     max_features=100,
                                     n_jobs=8, random_state=123,
                                     verbose=True)
    elif clf_name == "svc":
        from sklearn.svm import SVC
        clf = SVC(gamma='auto', random_state=123, verbose=True)

    elif clf_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(1029, 512, 128),
                            activation='relu',
                            early_stopping=True,
                            random_state=123)

    # Get the data from the Batcher
    x_train = batcher.train_data_encoded
    y_train = batcher.train_stances
    x_test = batcher.test_data_encoded
    y_test = batcher.test_stances
    target_names = batcher.stance_encoder.categories_[0]

    logger.debug("x-train: {} | "
                 "y-train: {}".format(x_train.shape, y_train.shape))
    logger.debug("x-test: {} | "
                 "y-test: {}".format(x_test.shape, y_test.shape))
    logger.debug("Target names: {}".format(target_names))

    # Train the classifier and predict
    clf.fit(x_train, y_train)
    y_pred = clf.predict(batcher.test_data_encoded)
    logger.debug("Predictions: {}".format(y_pred.shape))

    # Calculate score and clasificatin report
    score = clf.score(x_test, y_test)
    logger.info("Test score: {}".format(score))
    print(classification_report(y_test, y_pred, target_names=target_names))


def get_args():
    parser = argparse.ArgumentParser('StancePrediction task')

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

    # Encoder options
    encoder_group = parser.add_argument_group('Encoder Options')
    encoder_group.add_argument('--encoder', type=str,
                               default=("./external/models/LASER"
                                        "/bilstm.93langs.2018-12-26.pt"),
                               help='which encoder to be used')
    encoder_group.add_argument('--bpe-codes', type=str,
                               default=("./external/models/LASER/"
                                        "93langs.fcodes"),
                               help='Apply BPE using specified codes')
    encoder_group.add_argument('--vocab-file', type=str,
                               default=("./external/models/LASER/"
                                        "93langs.fvocab"),
                               help='Apply BPE using specified vocab')
    encoder_group.add_argument('--buffer-size', type=int, default=100,
                               help='Buffer size (sentences)')
    encoder_group.add_argument('--max-tokens', type=int, default=12000,
                               help='Max num tokens to process in a batch')
    encoder_group.add_argument('--max-sentences', type=int, default=None,
                               help='Max num sentences to process in a batch')
    encoder_group.add_argument('--cpu', action='store_true',
                               help='Use CPU instead of GPU')

    # Classifier options
    classifier_group = parser.add_argument_group('Classifier options')
    classifier_group.add_argument('--batch-size', type=int, default=32,
                                  help='Batch size to train the classifier')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    coloredlogs.install(fmt="%(levelname)s - %(filename)s - %(lineno)s: %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO,
                        logger=logger)



    # create preprocessor
    preprocessor = Preprocessor(args.workdir, sentence_encoder,
                                args.bpe_codes, args.vocab_file)

    # create batcher
    batcher = StanceBatcher(args.train_file, args.test_file,
                            args.batch_size, preprocessor.encode)

    prepare_save_or_load(batcher)

    make_classifier_and_classify(batcher)


