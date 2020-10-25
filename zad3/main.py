import numpy as np
import pandas as pd
import tensorflow.keras as kr
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.svm import SVC


def data_clear(data):
    data = data.dropna(how="any", axis=0)
    data = data.drop(columns=['track.id', 'track.name', 'track.artist', 'track.popularity', 'track.album.id',
                              'track.album.name', 'track.album.release_date', 'playlist_name', 'playlist_id',
                              'playlist_subgenre'])

    data['playlist_genre'] = data['playlist_genre'].map({'edm': 0, 'latin': 1, 'pop': 2, 'r&b': 3, 'rap': 4, 'rock': 5})

    # Data normalization to values 0-1
    data['key'] = data['key'] / 11
    data['duration_ms'] = (data['duration_ms'] - data['duration_ms'].min()) / (
            data['duration_ms'].max() - data['duration_ms'].min())
    data['loudness'] = (data['loudness'] - data['loudness'].min()) / (data['loudness'].max() - data['loudness'].min())
    data['tempo'] = (data['tempo'] - data['tempo'].min()) / (data['tempo'].max() - data['tempo'].min())
    return data


def delete_outliers(data):
    data.drop(data[data.tempo > 220].index, inplace=True)
    data.drop(data[data.tempo < 50].index, inplace=True)
    data.drop(data[data.key < 0.3].index, inplace=True)
    data.drop(data[data.loudness < -30].index, inplace=True)
    data.drop(data[data.speechiness > 0.8].index, inplace=True)
    data.drop(data[data.duration_ms > 1000000].index, inplace=True)

    return data


def nn(df_train, df_test):

    train, validate = np.split(df_train.sample(frac=1), [int(.8 * len(df_train))])

    train_y = pd.get_dummies(train['playlist_genre'])
    train_x = train.drop(columns='playlist_genre')
    val_y = pd.get_dummies(validate['playlist_genre'])
    val_x = validate.drop(columns='playlist_genre')
    test_y = df_test['playlist_genre']
    test_x = df_test.drop(columns='playlist_genre')

    model = kr.Sequential()

    # Neural network
    # model.add(kr.layers.Dense(20, input_dim=12, activation="sigmoid"))
    # model.add(kr.layers.Dense(10, activation="sigmoid"))
    # model.add(kr.layers.Dense(6, activation="sigmoid"))
    # model.summary()

    # L2 Regulalizer
    # model.add(kr.layers.Dense(100, input_dim=12, activation="sigmoid"))
    # model.add(kr.layers.Dense(50, activation="sigmoid", kernel_regularizer=kr.regularizers.L2(0.01)))
    # model.add(kr.layers.Dense(6, activation="sigmoid"))
    # model.summary()

    # DROPOUT
    # model.add(kr.layers.Dense(100, input_dim=12, activation="sigmoid"))
    # model.add(kr.layers.Dropout(0.2))
    # model.add(kr.layers.Dense(50, activation="sigmoid"))
    # model.add(kr.layers.Dropout(0.2))
    # model.add(kr.layers.Dense(6, activation="sigmoid"))
    # model.summary()

    # Batch normalization
    # model.add(kr.layers.Dense(30, input_dim=12, activation="sigmoid"))
    # model.add(kr.layers.Dense(10, activation="sigmoid"))
    # model.add(kr.layers.Dense(6, activation="sigmoid"))
    # model.summary()

    optimizer = kr.optimizers.Adam(learning_rate=0.01)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    early_stop = kr.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    training = model.fit(train_x, train_y, epochs=1000, validation_data=(val_x, val_y), callbacks=[early_stop])

    predicted = np.argmax(model.predict(test_x), axis=1)

    # Confusion matrix
    conf_m = tf.math.confusion_matrix(test_y, predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_m, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    # Summarize history for accuracy
    plt.plot(training.history['accuracy'])
    plt.plot(training.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def svm(train_df, test_df):

    train = train_df
    train_y = train['playlist_genre']
    train_x = train.drop(columns='playlist_genre')

    test_y = test_df['playlist_genre']
    test_x = test_df.drop(columns='playlist_genre')

    from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

    C_range = np.logspace(-2, 10, 6)

    param_grid = dict(C=C_range)

    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression()
    # print(lr.get_params().keys())

    # grid = GridSearchCV(SVC(verbose=1,), param_grid=param_grid, verbose=1)
    # grid.fit(train_x, train_y.values.ravel())
    # scores = grid.cv_results_['mean_test_score'].reshape(len(C_range))
    # print("Scores: ", scores)
    # print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    svc = SVC(C=2.5118864315095797)
    svc.fit(train_x, train_y.values.ravel())
    score = svc.score(test_x, test_y)
    print("Scores: ", score)


if __name__ == '__main__':
    data_test = pd.read_csv("test.csv")
    data_train = pd.read_csv("train.csv")

    data_train = delete_outliers(data_train)
    data_train = data_clear(data_train)

    data_test = data_clear(data_test)

    corr_matrix = data_train.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, vmax=1, square=True, annot=True, cmap='cubehelix')
    # plt.show()

    nn(data_train, data_test)
    # svm(data_train, data_test)
