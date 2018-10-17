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


def debug_info(df):
    logger.info('df = \n%s', df.describe(include='all'))
    for col in df:
        pylab.figure(col)
        pylab.hist(df[col])
        pylab.yscale('log')
    pylab.show()
    return


def evaluate(visitor_id, model, X):
    # this assumes X is sorted by visitor id
    logger.info('X.shape = %s', X.shape)
    lag_cols = []
    columns = list(X.columns)
    for i in range(1, 100):
        col = 'totals_transactionRevenue_lag%d' % i
        if col in columns:
            lag_cols.append(columns.index(col))
        else:
            break
    logger.info('lag found: %d', len(lag_cols))

    y_pred = []
    y = None
    vis = visitor_id[0]
    same_vis_count = 0
    for k in range(X.shape[0]):
        if k % 100000 == 0:
            logger.info('row index: %d', k)
        if vis == visitor_id[k]:  # only add lag for the same visitor
            for i in range(1, min(len(lag_cols), same_vis_count) + 1):
                X.iloc[k, lag_cols[i-1]] = y_pred[k-i]
        else:
            same_vis_count = 0
        X_tmp = np.array([X.values[k, :]])
        y = model.predict_on_batch(X_tmp).flatten()
        y_pred.append(y[0])
        vis = visitor_id[k]
        same_vis_count += 1

    return np.array(y_pred)


def main():
    df = load_predict_data()
    logger.info('column hash = %d', utils.column_hash(df))
    df = preprocess.drop_column(df, 'sessionId')

    visitor_id = df['fullVisitorId']
    df = preprocess.drop_column(df, 'fullVisitorId')

#    debug_info(df)

    trainings = load_trainings()

    logger.info('predicting')

    y_pred = None
    for training in trainings:
        try:
            logger.info('predicting with: ' + training['model_path'])
            model = training['model']
    #        quants = training['quants']
            scaler = training['scaler']
            X = preprocess.drop_column(df, 'totals_transactionRevenue')
            columns = X.columns
            logger.info('X.shape = %s', X.shape)
            X = scaler.transform(X)
    #         y_classes = model.predict(X)
    #         y_tmp = postprocess.make_real_predictions(y_classes, quants)
            y_tmp = evaluate(visitor_id, model, pd.DataFrame(X, columns=columns))
            if y_pred is None:
                y_pred = y_tmp
            else:
                y_pred = np.add(y_pred, y_tmp)
        except KeyboardInterrupt:
            break

    y_pred /= len(trainings)

    logger.info('y_pred.shape = %s', y_pred.shape)
    save_prediction(visitor_id, y_pred)

    hist_revenue(y_pred, y_pred.max())
    pylab.show()


if __name__ == '__main__':
    utils.setup_logging()
    main()
