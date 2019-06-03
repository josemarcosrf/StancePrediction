import csv
import logging
import coloredlogs
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# configure the logger
coloredlogs.install(logger=logger, level=logging.INFO,
                    format="%(filename)s:%(lineno)s - %(message)s")


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


class StanceDataLoader():

    VALID_KEYS = {'Tweet', 'Target', 'Stance'}
    VALID_SETS = {'train', 'test'}

    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self._load_dataset()

    def _load_dataset(self):
        """ Loads train and test dataset from CSV files """
        try:
            # Trainig data
            logger.info(
                "Loading training data from: {}".format(self.train_file))
            self.train_df = load_from_csv(self.train_file)
            logger.info("Train targets count:\n{}".format(
                self.train_df['Target'].value_counts())
            )

            # Test data
            logger.info(
                "Loading testing data from: {}".format(self.test_file))
            self.test_df = load_from_txt(self.test_file)
            logger.info("Test targets count:\n{}".format(
                self.test_df['Target'].value_counts())
            )
        except Exception as e:
            logger.error("Error while reading Stance dataset!")
            logger.exception(e)

    def get(self, key, set='train'):
        if key not in self.VALID_KEYS:
            raise ValueError("{} is not a valid dataset field. "
                             "Valid fields: {}".format(key, self.VALID_KEYS))

        if set not in self.VALID_SETS:
            raise ValueError("{} is not a valid set. "
                             "Must be one of {}".format(key, self.VALID_SETS))

        if set == 'train':
            return self.train_df[key].tolist()

        return self.test_df[key].tolist()

