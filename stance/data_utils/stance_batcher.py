import logging
import coloredlogs

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from stance.data_utils import (load_from_csv, load_from_txt,
                               to_tensor, chunk)

logger = logging.getLogger(__name__)

# configure the logger
coloredlogs.install(logger=logger, level=logging.INFO,
                    format="%(filename)s:%(lineno)s - %(message)s")


class StanceBatcher():

    def __init__(self, train_file, test_file, batch_size, tweet_preproc_func):
        """ Loads, Transforms and batches the Stance dataset preparing it
        for training

        Args:
            train_file (str): CSV file containing training examples
            test_file (str): CSV file containing test examples
            batch_size (int): size of each batch
            preprocessor (function):
                Sentence preprocessor function to be called for each input
        """
        self.tweet_preproc_func = tweet_preproc_func
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size

        # Encoders
        self.target_encoder = OneHotEncoder(handle_unknown='ignore')
        self.stance_encoder = OneHotEncoder(handle_unknown='ignore')

        # Attributes to be populated by '_load_dataset':
        self.train_df = None
        self.test_df = None
        self.train_data_encoded = None
        self.test_data_encoded = None
        self._load()

    def _load(self):
        try:
            self._load_datasets()
        except Exception as e:
            logger.error("Error while reading Stance dataset!")
            logger.exception(e)

    def _load_datasets(self):
        """ Loads train and test dataset from CSV files """
        # Trainig data
        logger.info("Loading training"
                    " data from: {}".format(self.train_file))
        self.train_df = load_from_csv(self.train_file)

        logger.info("Train targets count:\n{}".format(
            self.train_df['Target'].value_counts())
        )
        # Test data
        logger.info("Loading testing"
                    " data from: {}".format(self.test_file))
        self.test_df = load_from_txt(self.test_file)

        logger.info("Test targets count:\n{}".format(
            self.test_df['Target'].value_counts())
        )

    def prepare_inputs(self):
        """ Applies text preprocesing to the inptus with the given
        'tweet_preproc_func' function
        """
        try:
            # encode train data
            train_input_text = self.train_df['Tweet'].tolist()
            self.train_data_encoded = self.tweet_preproc_func(train_input_text)
            logger.info("Train data encoded: {}"
                        "".format(self.train_data_encoded.shape))
            # encode test data
            test_input_text = self.test_df['Tweet'].tolist()
            self.test_data_encoded = self.tweet_preproc_func(test_input_text)
            logger.info("Test data encoded: {}"
                        "".format(self.test_data_encoded.shape))

            # Concatenate encoded inputs and encoded targets
            train_targets, test_targets = self._encode_targets()
            self.train_data_encoded = np.concatenate((self.train_data_encoded,
                                                      train_targets), axis=1)
            self.test_data_encoded = np.concatenate((self.test_data_encoded,
                                                     test_targets), axis=1)

        except Exception as e:
            logger.error("Error while preparing inputs!")
            logger.exception(e)

    def prepare_outputs(self):
        # Encode stance labels
        self.train_stances, self.test_stances = self._encode_stance()
        self.train_stances = np.argmax(self.train_stances, axis=1)
        self.test_stances = np.argmax(self.test_stances, axis=1)

    def _encode_targets(self):
        # Fit the one-hot encoder on all targets
        train_targets = np.array(self.train_df['Target'].tolist()).reshape(-1, 1)
        test_targets = np.array(self.test_df['Target'].tolist()).reshape(-1, 1)

        self.target_encoder.fit(np.concatenate(
            (train_targets, test_targets), axis=0))

        return (self.target_encoder.transform(train_targets).todense(),
                self.target_encoder.transform(test_targets).todense())

    def _encode_stance(self):
        train_stances = np.array(self.train_df['Stance'].tolist()).reshape(-1, 1)
        test_stances = np.array(self.test_df['Stance'].tolist()).reshape(-1, 1)

        self.stance_encoder.fit(np.concatenate(
            (train_stances, test_stances), axis=0))

        return (self.stance_encoder.transform(train_stances).todense(),
                self.stance_encoder.transform(test_stances).todense())

    def load_encoded_inputs(self, training_mat_path, test_mat_path):
        """Loads from file train and test numpy matrices of encoded inputs
        (N x 1024) each.
        Where N is the number of training or testing samples.

        Args:
            training_mat_path (str): train encoded inputs matrix file path
            test_mat_path (str): test encoded inputs matrix file path
        """
        try:
            logger.info("Loading encoded training matrix"
                        " from: {}".format(training_mat_path))
            self.train_data_encoded = np.load(training_mat_path)

            logger.info("Loading encoded test matrix"
                        " from: {}".format(test_mat_path))
            self.test_data_encoded = np.load(test_mat_path)
        except Exception as e:
            logger.error("Error loading encoded inputs as numpy file")
            logger.exception(e)

    def get_training_batches(self):
        """Returns an iterator yielding a chunk (batch) of training
        tuples containing inputs and targets.

        Args:
            batch_size (int): size of each batch
        """
        return self._get_batches(self.train_data_encoded,
                                 self.train_stances,
                                 self.batch_size)

    def get_val_batch(self):
        """Returns an iterator yielding a chunk (batch) of evaluations
        tuples containing inputs and targets.

        Args:
            batch_size (int): size of each batch
        """
        return self._get_batches(self.test_data_encoded,
                                 self.test_stances,
                                 self.batch_size)

    def _get_batches(self, inputs, outputs, batch_size):
        # convert data to torch.Tensor before batching
        inputs = to_tensor(inputs)
        outputs = to_tensor(outputs)
        logger.debug("{} input batches available".format(inputs.shape))
        logger.debug("{} output batches available".format(outputs.shape))

        return zip(
            chunk(inputs, batch_size),
            chunk(outputs, batch_size)
        )
