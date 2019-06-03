import logging
import coloredlogs
from stance.data_utils import (to_tensor, chunk)

logger = logging.getLogger(__name__)

# configure the logger
coloredlogs.install(logger=logger, level=logging.INFO,
                    format="%(filename)s:%(lineno)s - %(message)s")


class StanceBatcher():

    def __init__(self, x_train, y_train, x_val, y_val, batch_size):
        """Creates tuple iterators to yield X and Y pairs for training
        and evaluation of pyTorch models.

        Args:
            x_train ([type]): [description]
            y_train ([type]): [description]
            x_val ([type]): [description]
            y_val ([type]): [description]
            batch_size ([type]): [description]
        """
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def get_training_batches(self):
        """Returns an iterator yielding a chunk (batch) of training
        tuples containing inputs and targets.

        Args:
            batch_size (int): size of each batch
        """
        return self._get_batches(self.x_train,
                                 self.y_train,
                                 self.batch_size)

    def get_val_batch(self):
        """Returns an iterator yielding a chunk (batch) of evaluations
        tuples containing inputs and targets.

        Args:
            batch_size (int): size of each batch
        """
        return self._get_batches(self.x_val,
                                 self.y_val,
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
