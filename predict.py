import pandas as pd
import preprocess
import postprocess
import logging
import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics.regression import mean_squared_error
import pylab
import utils


logger = logging.getLogger(__name__)


def load_predict_data():
    logger.info('loading predict data')
    return preprocess.load_data('test')


def load_trainings():
    logger.info('loading trainings')
    filenames = []
    for f in os.listdir('model'):
        if f.startswith('training'):
            filenames.append(os.path.join('model', f))
    trainings = []
    for filename in filenames:
        with open(filename, 'rb') as fi:
            loaded = pickle.load(fi)
            loaded['model'] = tf.keras.models.load_model(loaded['model_path'])
            trainings.append(loaded)
    return trainings


def save_prediction(visitor_id, y_pred):
    logger.info('saving prediction')
    df = pd.DataFrame()
    df['fullVisitorId'] = visitor_id
    df['totals_transactionRevenue'] = y_pred
    df.to_csv('prediction.csv', index=False)


def hist_revenue(y, y_max, name=None):
    pylab.figure(name)
    pylab.hist(y, bins=20, range=[0, y_max])
    pylab.yscale('log')
    pylab.ylim([0.5, 1e7])


def main():
    df = load_predict_data()

    visitor_id = df['fullVisitorId']
    df = preprocess.drop_column(df, 'fullVisitorId')

    trainings = load_trainings()

    logger.info('predicting')

    y_pred = None
    for training in trainings:
        logger.info('predicting with: ' + training['model_path'])
        model = training['model']
        scaler = training['scaler']
        X = preprocess.drop_column(df, 'totals_transactionRevenue')
        logger.info('X.shape = %s', X.shape)
        X = scaler.transform(X)
        if y_pred is None:
            y_pred = model.predict(X).flatten()
        else:
            y_pred = np.add(y_pred, model.predict(X).flatten())

    y_pred /= len(trainings)
    y_pred = np.clip(y_pred, 0, 100)

    logger.info('y_pred.shape = %s', y_pred.shape)
    save_prediction(visitor_id, y_pred)

    hist_revenue(y_pred, y_pred.max())
    pylab.show()


if __name__ == '__main__':
    utils.setup_logging()
    main()
