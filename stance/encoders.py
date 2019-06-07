import logging
import coloredlogs

import regex as re
import numpy as np

from stance.utils.text_processing import tokenize
# TODO: FIX the pyBPE path mess!
from external.pyBPE.pybpe.pybpe import pyBPE

logger = logging.getLogger(__name__)


# configure the logger
coloredlogs.install(logger=logger, level=logging.INFO,
                    format="%(filename)s:%(lineno)s - %(message)s")


class OneHotLabelEncoder():

    def __init__(self, x):
        from sklearn.preprocessing import OneHotEncoder

        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)

        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(x)

    def encode(self, x):
        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)
        return self.encoder.transform(x).todense()

    def get_labels(self):
        return self.encoder.categories_[0]


class IndexEncoder():

    def __init__(self, x):
        from sklearn import preprocessing

        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)

        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(x)

    def encode(self, x):
        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)
        return self.encoder.transform(x)

    def get_labels(self):
        return self.encoder.classes_


class LaserEncoder():

    """LaserEncoder for the Stance dataset.
    Given a batch of sentences encode the batch as a matrix (N x E)
    using Language Agnostic SEntence Representations:
    https://arxiv.org/abs/1812.10464
    """

    HANDLER_REGEX = r"@[\w\d_-]+"

    def __init__(self, args, lang='en', lower_case=False, descape=False):
        """Text corpus encoder using LASER: https://arxiv.org/abs/1812.10464

        Args:
            args ([type]): [description]
            lang (str, optional): [description]. Defaults to 'en'.
            lower_case (bool, optional): [description]. Defaults to False.
            descape (bool, optional): [description]. Defaults to False.
        """

        # TODO: Currently the external Laser lib. requires an args object.
        #       Change for explicit parameters!
        from external.encoders.laser import EncodeLoad

        # configure path from 'args'
        self.workdir = args.workdir
        self.bpe_codes_file = args.bpe_codes
        self.vocab_file = args.vocab_file

        # BPE encoding
        self.bpe = pyBPE(self.vocab_file, self.bpe_codes_file)
        self.bpe.load()

        # load the LASER sentence encoder
        self.encoder = EncodeLoad(args)

        # tokenization configuration
        self.lang = lang
        self.lower_case = lower_case
        self.descape = descape

    def encode(self, corpus):
        """Encodes a given corpus using LASER encoding.

        Args:
            corpus (iterable): of strings composing the text corpus to encde
        """
        # preprocess all corpus examples:
        # NOTE: Very suboptimal as we load the codes and vocab for each example
        preproc_questions = [self._preproc(text) for text in corpus]

        # LASER encode
        mat = self.encoder.encode_sentences(preproc_questions)
        return mat

    def _preproc(self, input_text):

        # Remove twitter handlers
        input_text = re.sub(self.HANDLER_REGEX,
                            "TWEETER_HANDLER", input_text)
        # Tokenize the input
        tokenized = tokenize(input_text, lang=self.lang,
                             descape=False, lower_case=False)
        # BPE encode
        encoded = self.bpe.apply_bpe(tokenized)
        return encoded
