import logging
import coloredlogs

import regex as re

from stance.data_utils.text_processing import tokenize

logger = logging.getLogger(__name__)


# configure the logger
coloredlogs.install(logger=logger, level=logging.INFO,
                    format="%(filename)s:%(lineno)s - %(message)s")


class LaserEncoder():

    """LaserEncoder for the Stance dataset.
    Given a batch of sentences encode the batch as a matrix (N x E)
    using Language Agnostic SEntence Representations:
    https://arxiv.org/abs/1812.10464
    """

    HANDLER_REGEX = r"@[\w\d_-]+"

    def __init__(self, workdir, args,
                 bpe_codes_file, vocab_file,
                 lang='en', lower_case=False, descape=False):

        # FIX this mess of paths!
        from external.encoders.laser import EncodeLoad
        from external.pyBPE.pybpe.pybpe import pyBPE as bpe

        # load the LASER sentence encoder
        self.encoder = EncodeLoad(args)
        self.bpe_codes_file = bpe_codes_file
        self.vocab_file = vocab_file

        # configuration
        self.lang = lang
        self.lower_case = lower_case
        self.descape = descape
        self.workdir = workdir

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
        encoded = bpe.apply_bpe(tokenized,
                                self.bpe_codes_file,
                                self.vocab_file)

        return encoded
