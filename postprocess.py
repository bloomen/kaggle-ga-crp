import numpy as np


def make_real_predictions(y_classes, y_max, shuffle=True):
    n_classes = y_classes.shape[1]
    step = y_max / n_classes
    y_pred = []
    for row in y_classes:
        value = np.argmax(row) * step
        if shuffle and value > 0:
            value += np.random.random() * step
        y_pred.append(value)
    return np.array(y_pred)
