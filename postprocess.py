import numpy as np
import logging

logger = logging.getLogger(__name__)


def make_real_predictions(y_classes, quants):
    logger.info("making real predictions")
    logger.info("y_classes.shape = %s", y_classes.shape)
    logger.info("quants.shape = %s", quants.shape)
    y_pred = []
    for row in y_classes:
        value = np.argmax(row)
        if value > 0:
            value = (quants[value-1] + quants[value]) / 2
        y_pred.append(value)
    y_pred = np.array(y_pred)
    logger.info("y_pred.shape = %s", y_pred.shape)
    return y_pred


def make_real_predictions2(y_classes, y_max, shuffle=True):
    n_classes = y_classes.shape[1]
    step = y_max / n_classes
    y_pred = []
    for row in y_classes:
        value = np.argmax(row) * step
        if shuffle and value > 0:
            value += np.random.random() * step
        y_pred.append(value)
    return np.array(y_pred)
