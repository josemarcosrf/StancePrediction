import os
import coloredlogs
import logging

import numpy as np

from stance import STANCES, TARGETS
from stance import (timeit, build_arg_parser)
from stance.data_utils.loaders import SemEvalDataLoader
from stance.encoders import (LaserEncoder,
                             OneHotLabelEncoder,
                             IndexEncoder)


logger = logging.getLogger(__name__)


def encode_tweets_and_targets(args, tweets, targets):
    """ Applies text preprocesing and LASER encodes the
    tweet entries. Applies OneHot encoding to the targets """
    try:

        # Encode targets as OneHot vectors
        target_encoder = OneHotLabelEncoder(TARGETS)
        encoded_targets = target_encoder.encode(targets)
        logger.debug("Targets found: {}".format(target_encoder.get_labels()))

        # encode Tweet data
        encoder = LaserEncoder(args)
        encoded_tweets = encoder.encode(tweets)
        logger.info("Tweets encoded: {}".format(encoded_tweets.shape))

        # Concatenate encoded tweets and encoded targets
        return np.concatenate((encoded_tweets, encoded_targets), axis=1)

    except Exception as e:
        logger.error("Error while preparing inputs!")
        logger.exception(e)


def encode_outputs(stances):
    """Encode stance labels as numeric indices

    Args:
        stances (list): list of stances labels

    Returns:
        tuple(list, list): list of numeric encoded stances and seen clas labels
    """
    encoder = IndexEncoder(STANCES)
    return encoder.encode(stances), encoder.get_labels()


def load_encoded_inputs(training_mat_path, test_mat_path):
    """Loads from file train and test numpy matrices of encoded inputs
    (N x 1024+len(targets)) each.
    Where N is the number of training or testing samples.

    Args:
        training_mat_path (str): train encoded inputs matrix file path
        test_mat_path (str): test encoded inputs matrix file path
    """
    try:
        logger.info("Loading encoded training matrix"
                    " from: {}".format(training_mat_path))
        train_data_encoded = np.load(training_mat_path)

        logger.info("Loading encoded test matrix"
                    " from: {}".format(test_mat_path))
        test_data_encoded = np.load(test_mat_path)

        return train_data_encoded, test_data_encoded
    except Exception as e:
        logger.error("Error loading encoded inputs as numpy file")
        logger.exception(e)


@timeit
def encode_or_load_data(args, data_loader):
    """Encodes or loads the Stance dataset

    Args:
        args ([type]): [description]
        data_loader ([type]): [description]

    Returns:
        [type]: [description]
    """
    # ** Inputs **
    encoded_train_inputs_path = os.path.join(
        args.workdir, "training_laser-tweet+onehot-target.npy")
    encoded_test_inputs_path = os.path.join(
        args.workdir, "test_laser-tweet+onehot-target.npy")

    if not os.path.exists(encoded_train_inputs_path) or \
            not os.path.exists(encoded_test_inputs_path):

        # Transform and save if not present
        if not os.path.exists(encoded_train_inputs_path):
            try:
                # Preprocess and encode the inputs (Tweets + Targets)
                encoded_training_inputs = encode_tweets_and_targets(
                    args,
                    data_loader.get('Tweet', set='train'),
                    data_loader.get('Target', set='train')
                )
                np.save(encoded_train_inputs_path, encoded_training_inputs)
            except Exception as e:
                logger.error("Error while encoding train dataset")
                logger.exception(e)

        if not os.path.exists(encoded_test_inputs_path):
            try:
                encoded_test_inputs = encode_tweets_and_targets(
                    args,
                    data_loader.get('Tweet', set='test'),
                    data_loader.get('Target', set='test')
                )
                np.save(encoded_test_inputs_path, encoded_test_inputs)
            except Exception as e:
                logger.error("Error while encoding test dataset")
                logger.exception(e)

    else:
        encoded_training_inputs, encoded_test_inputs = \
            load_encoded_inputs(encoded_train_inputs_path,
                                encoded_test_inputs_path)

    # ** Outputs **
    train_outputs, _ = encode_outputs(data_loader.get('Stance', set='train'))
    test_outputs, lbls = encode_outputs(data_loader.get('Stance', set='test'))

    return ((encoded_training_inputs, train_outputs),
            (encoded_test_inputs, test_outputs),
            lbls)


def save_model(args, clf):
    from sklearn.externals import joblib

    save_path = os.path.join(
        args.workdir, "{}_{}.pkl".format(args.clf_save_name, args.clf_type))
    joblib.dump(clf, save_path)


def load_model(args):
    from sklearn.externals import joblib

    save_path = os.path.join(
        args.workdir, "{}_{}.pkl".format(args.clf_save_name, args.clf_type))
    return joblib.load(save_path)


def save_predictions(args, y_pred):
    from stance.data_utils.loaders import load_from_txt
    import copy

    gold = load_from_txt(args.test_file)[['Target', 'Tweet', 'Stance']]
    pred = copy.deepcopy(gold)

    for i, y in enumerate(y_pred):
        y_lbl = ['AGAINST', 'FAVOR', 'NONE'][y]
        pred.iloc[i]['Stance'] = y_lbl

    pred.to_csv(os.path.join(args.workdir, "predictions.csv"),
                sep='\t', index=True, index_label='ID')


def eval_model(args, clf, x_test, y_test):
    from sklearn.metrics import classification_report

    y_pred = clf.predict(x_test)
    logger.debug("Predictions: {}".format(y_pred.shape))

    # Calculate score and clasificatin report
    score = clf.score(x_test, y_test)
    logger.info("Test score: {}".format(score))
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save predictions as csv
    save_predictions(args, y_pred)


@timeit
def make_classifier_and_predict(args, train_set, test_set,
                                target_names, random_seed,
                                clf_name="randomforest"):
    # Data
    x_train, y_train = train_set
    x_test, y_test = test_set

    logger.debug("x-train: {} | "
                 "y-train: {}".format(x_train.shape, y_train.shape))
    logger.debug("x-test: {} | "
                 "y-test: {}".format(x_test.shape, y_test.shape))
    logger.debug("Target names: {}".format(target_names))

    # Out-of-the-box Classifiers
    if clf_name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=100, n_estimators=1000,
                                     max_features=100,
                                     n_jobs=8, random_state=random_seed,
                                     verbose=True)
    elif clf_name == "svc":
        from sklearn.svm import SVC
        clf = SVC(gamma='auto', random_state=random_seed, verbose=True)

    elif clf_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(1029, 512, 128),
                            activation='relu',
                            early_stopping=True,
                            random_state=random_seed)

    # Train the classifier & save
    clf.fit(x_train, y_train)
    save_model(args, clf)
    eval_model(args, clf, x_test, y_test)
    return clf


@timeit
def encode_or_load_stance(args, data_loader):

    encoded_stance_input_file = "./workdir/stance_laser-text+onehot-target.npy"

    if not os.path.exists(encoded_stance_input_file):
        try:
            encoded_inputs = encode_tweets_and_targets(
                args,
                [t.replace("\n", "  ") for t in data_loader.get('text')],
                data_loader.get('controversial trending issue')
            )
            np.save(encoded_stance_input_file, encoded_inputs)
        except Exception as e:
            logger.error("Error while encoding stance inputs...")
            logger.exception(e)
    else:
        encoded_inputs = np.load(encoded_stance_input_file)

    return encoded_inputs


def get_args():
    parser = build_arg_parser()

    # Add encoder options
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

    # Add classifier options
    classifier_group = parser.add_argument_group('Classifier options')
    classifier_group.add_argument('--clf-save-name', default='laser',
                                  help="Name to save the classifier")
    classifier_group.add_argument('--clf-type', default='mlp',
                                  help=('Classifer type to use: '
                                        '{mlp, randomforest, svc}'))
    classifier_group.add_argument('--batch-size', type=int, default=32,
                                  help='Batch size to train the classifier')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    coloredlogs.install(
        fmt="%(levelname)s - %(filename)s - %(lineno)s: %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
        logger=logger
    )

    if args.task in {'train', 'eval'}:
        # Load dataset
        data_loader = SemEvalDataLoader(args.train_file, args.test_file)
        # Create preprocessor
        train_set, test_set, target_names = encode_or_load_data(args, data_loader)

    # Train a classifier and evaluate on the test set
    if args.task == 'train':
        clf = make_classifier_and_predict(args,
                                          train_set, test_set,
                                          target_names,
                                          args.rseed,
                                          args.clf_type)

    # Load a trained clasifier and evaluate on the test set
    elif args.task == 'eval':
        clf = load_model(args)
        x_test, y_test = test_set
        eval_model(args, clf, x_test, y_test)

    elif args.task == 'transfer':
        from stance.data_utils.loaders import StanceDataLoader

        # load the stance.csv dataset....
        data_loader = StanceDataLoader(args.test_file)
        x = encode_or_load_stance(args, data_loader)

        # load the model & eval
        clf = load_model(args)
        y = clf.predict(x)

        # print some result for visual inspection
        texts = data_loader.get('text')
        issues = data_loader.get('controversial trending issue')
        for t, issue, pred in zip(texts, issues, y):
            pred_lbl = ['AGAINST', 'FAVOR', 'NONE'][pred]
            print("{} | {} --> {}".format(t, issue, pred_lbl))
            input("....")

    else:
        logger.error("Task: {} is not a valid task. "
                     "Valid tasks are: {train,eval}".format(args.task))
