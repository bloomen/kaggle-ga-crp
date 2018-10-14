import logging
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(name)s:%(lineno)s - %(funcName)s - %(levelname)s - %(message)s', 
        level=logging.DEBUG
    )


def split_data(X, y, ratio=0.2, seed=0):
    logger.info('splitting data')
    if seed == 0:
        seed = int(time.time())
    return train_test_split(X, y, test_size=ratio, random_state=seed)


class EpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log):
        logger.info("epoch = %d / val_loss = %f", epoch, log['val_loss'])


def column_hash(df):
    col_names = '-'.join(df.columns)
    total = 0
    for i, c in enumerate(col_names):
        if i % 2 == 0:
            total += ord(c)
        else:
            total -= ord(c)
    return total
