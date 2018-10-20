import utils
import logging
import preprocess
import postprocess
import pylab
from sklearn.preprocessing.data import StandardScaler, PolynomialFeatures,\
    MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics.regression import mean_squared_error
import tensorflow as tf
import os
import pickle
from sklearn.decomposition.pca import PCA
from sklearn.linear_model.base import LinearRegression


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


def build_regressor(n_features):
#    np.random.seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2*n_features, activation=tf.nn.relu, input_shape=(n_features,)),
        tf.keras.layers.Dense(2*n_features, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse')
    return model


def build_classifier(n_features, n_classes):
#    np.random.seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(n_features,)),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def hist_revenue(y, name=None):
    pylab.figure(name)
    pylab.hist(y, bins=20, range=[0, y.max()])
    pylab.yscale('log')
    pylab.ylim([0.5, 1e7])


def save_model(model, i, quants, scaler):
    dir_name = 'model'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    filename = os.path.join(dir_name, 'model_%04d.h5' % i)
    logger.info('saving model to: %s', filename)
    tf.keras.models.save_model(model, filename)
    metadata = {
        'model_path': filename,
        'quants': quants,
        'scaler': scaler,
    }
    filename_meta = os.path.join(dir_name, 'training_%04d.pickle' % i)
    logger.info('saving metadata to: %s', filename_meta)
    with open(filename_meta, 'wb') as f:
        pickle.dump(metadata, f)


def save_model2(model, linear_model, i, y_max, poly, scaler):
    dir_name = 'model'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    filename = os.path.join(dir_name, 'model_%04d.h5' % i)
    logger.info('saving model to: %s', filename)
    tf.keras.models.save_model(model, filename)
    metadata = {
        'model_path': filename,
        'linear_model': linear_model,
        'y_max': y_max,
        'poly': poly,
        'scaler': scaler,
    }
    filename_meta = os.path.join(dir_name, 'training_%04d.pickle' % i)
    logger.info('saving metadata to: %s', filename_meta)
    with open(filename_meta, 'wb') as f:
        pickle.dump(metadata, f)


def plot_history_classifier(history):
    logger.info(history.history.keys())
    pylab.figure()
    pylab.xlabel('Epoch')
    pylab.ylabel('Accuracy')
    pylab.plot(history.epoch, np.sqrt(np.array(history.history['acc'])),
               label='Train')
    pylab.plot(history.epoch, np.sqrt(np.array(history.history['val_acc'])),
               label='Validation')
    pylab.legend()
    pylab.ylim([0.9, 1])


def plot_history_regressor(history):
    logger.info(history.history.keys())
    pylab.figure()
    pylab.xlabel('Epoch')
    pylab.ylabel('Loss')
    pylab.plot(history.epoch, np.sqrt(np.array(history.history['loss'])),
               label='Train')
    pylab.plot(history.epoch, np.sqrt(np.array(history.history['val_loss'])),
               label='Validation')
    pylab.legend()
    pylab.ylim([0, 3])


def add_poly_features(poly, X, fit=False):
    logger.info('adding polynomial features: fit=%s', fit)
    poly_train_df = X[preprocess.poly_features(X)]
    if fit:
        poly_train = poly.fit_transform(poly_train_df.values)
    else:
        poly_train = poly.transform(poly_train_df.values)
    poly_train = pd.DataFrame(poly_train,
                              columns=poly.get_feature_names())
    poly_train['sessionId'] = X.index
    poly_train.set_index('sessionId')
    X = X.merge(poly_train, on='sessionId', how='left')
    X = preprocess.drop_column(X, 'sessionId')
    logger.info('X.shape = %s', X.shape)
    return X


def main():
    df = load_train_data()
    logger.info('column hash = %d', utils.column_hash(df))
    df = preprocess.drop_column(df, 'fullVisitorId')
    df = preprocess.drop_column(df, 'visitStartTime')
#    df = preprocess.drop_column(df, 'sessionId')
#    debug_info(df)

    y = df['totals_transactionRevenue']
    X = preprocess.drop_column(df, 'totals_transactionRevenue')

#    X, _, y, _ = utils.split_data(X, y, ratio=0.9, seed=42)

#    n_classes = 10
    n_models = 1

    y_max = y.max()

    for i in range(n_models):

        X_train, X_test, y_train, y_test = utils.split_data(X, y)

        logger.info('training')

#         y_train, quants = preprocess.make_class_target(y_train, n_classes)
#         logger.info('y_train.unique() = %s', y_train.unique())
#         logger.info('quants = %s', quants)

#        y_train = preprocess.make_class_target2(y_train, y_max, n_classes)

        poly = PolynomialFeatures(degree=2)

        X_train = add_poly_features(poly, X_train, fit=True)

        scaler = StandardScaler()
#        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)

        pca = PCA(n_components=30)
        X_train = pca.fit_transform(X_train)

        logger.info('X_train.shape = %s', X_train.shape)

#         cumulative = np.cumsum(pca.explained_variance_ratio_)
#         pylab.plot(cumulative, 'r-')
#         pylab.show()

#        model = build_classifier(X_train.shape[1], n_classes)
        model = build_regressor(X_train.shape[1])
        EPOCHS = 100
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5)
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, utils.EpochCallback()])

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        logger.info('predicting')
        logger.info('X_test.shape = %s', X_test.shape)

        poly_test = X_test[preprocess.poly_features(X_test)]
        poly_test = pd.DataFrame(poly.transform(poly_test), columns=poly_test.columns)
        X_test = pd.concat([X_test, poly_test], axis=1)

        X_test = add_poly_features(poly, X_test)

        X_test = scaler.transform(X_test)
        X_test = pca.transform(X_test)

#        y_classes = model.predict(X_test)
#        y_pred = postprocess.make_real_predictions(y_classes, quants)
#        y_pred = postprocess.make_real_predictions2(y_classes, y_max)

        y_pred = model.predict(X_test).flatten()
        y_linear_pred = linear_model.predict(X_test)

        rms = np.sqrt(mean_squared_error(y_test, y_pred))
        linear_rms = np.sqrt(mean_squared_error(y_test, y_linear_pred))
        logger.info('rms = %s', rms)
        logger.info('linear_rms = %s', linear_rms)

#        save_model(model, i, quants, scaler)
        save_model2(model, linear_model, i, y_max, poly, scaler)

#    plot_history_classifier(history)
    plot_history_regressor(history)

    pylab.figure()
    pylab.scatter(y_pred, y_test, alpha=0.5)
    pylab.xlabel("pred")
    pylab.ylabel("test")

    hist_revenue(y_linear_pred, 'y_linear_pred')
    hist_revenue(y_pred, 'y_pred')
    hist_revenue(y_test, 'y_test')

    pylab.show()


if __name__ == '__main__':
    utils.setup_logging()
    main()
