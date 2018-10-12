import utils
import logging
import preprocess
import pylab
from sklearn.preprocessing.data import StandardScaler
from sklearn.linear_model.base import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics.regression import mean_squared_error
import tensorflow as tf
import os
import pickle


logger = logging.getLogger(__name__)


def load_train_data():
    logger.info('loading train data')
    return preprocess.load_data('train')


def debug_info(df):
    logger.info('df = \n%s', df.describe(include='all'))
    for col in df:
        pylab.figure(col)
        pylab.hist(df[col])
        pylab.yscale('log')
    pylab.show()
    return


def make_log_revenue(visitor_id, revenue):
    # TODO: do this in C++ later
    logger.info('making log revenue')
    df = pd.DataFrame()
    df['fullVisitorId'] = visitor_id
    df['PredictedLogRevenue'] = revenue
    gb = df.groupby('fullVisitorId')
    logger.info('groupby size = %d', len(gb))

    def reduce(group):
        revenue = group['PredictedLogRevenue']
        revenue = np.exp(revenue) - 1
        return np.log(revenue.sum() + 1)

    df = gb.apply(reduce)
    logger.info('df.shape = %s', df.shape)
    return df


def build_model(n_features):
    np.random.seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(n_features,)),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


def save_model(model, i, scaler):
    dir_name = 'model'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    filename = os.path.join(dir_name, 'model_%04d.h5' % i)
    logger.info('saving model to: %s', filename)
    tf.keras.models.save_model(model, filename)
    metadata = {
        'model_path': filename,
        'scaler': scaler,
    }
    filename_meta = os.path.join(dir_name, 'training_%04d.pickle' % i)
    logger.info('saving metadata to: %s', filename_meta)
    with open(filename_meta, 'wb') as f:
        pickle.dump(metadata, f)


def main():
    df = load_train_data()
    df = preprocess.drop_column(df, 'fullVisitorId')

    y = df['totals_transactionRevenue']
    X = preprocess.drop_column(df, 'totals_transactionRevenue')
    X_train, X_test, y_train, y_test = utils.split_data(X, y, seed=42)

    logger.info('training')
    logger.info('X_train.shape = %s', X_train.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = build_model(X_train.shape[1])
    EPOCHS = 100
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    model.fit(X_train, y_train, epochs=EPOCHS,
              validation_split=0.2, verbose=0,
              callbacks=[early_stop, utils.EpochCallback()])

    logger.info('predicting')
    logger.info('X_test.shape = %s', X_test.shape)

    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test).flatten()

    y_pred = np.clip(y_pred, 0, 100)

    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info('rms = %s', rms)

    save_model(model, 0, scaler)

    pylab.figure()
    pylab.scatter(y_pred, y_test, alpha=0.5)
    pylab.xlabel("pred")
    pylab.ylabel("test")

    pylab.figure('pred')
    pylab.hist(y_pred)
    pylab.yscale('log')

    pylab.figure('test')
    pylab.hist(y_test)
    pylab.yscale('log')

    pylab.show()


if __name__ == '__main__':
    utils.setup_logging()
    main()
