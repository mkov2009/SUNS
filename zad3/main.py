import numpy as np
import pandas as pd
import tensorflow.keras as kr
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def data_clear(df):
    df = df.dropna(how="any", axis=0)
    df = df.drop(columns=['track.id', 'track.name', 'track.artist', 'track.popularity', 'track.album.id',
                          'track.album.name', 'track.album.release_date', 'playlist_name', 'playlist_id',
                          'playlist_subgenre'])
    df['playlist_genre'] = df['playlist_genre'].map({'edm': 0, 'latin': 1, 'pop': 2, 'r&b': 3, 'rap': 4, 'rock': 5})
    df['key'] = df['key'] / 11

    # Data normalization to values 0-1
    df['duration_ms'] = (df['duration_ms'] - df['duration_ms'].min()) / (df['duration_ms'].max() - df['duration_ms'].min())
    df['loudness'] = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())
    df['tempo'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())

    return df


if __name__ == '__main__':
    data_test = pd.read_csv("test.csv")
    data_train = pd.read_csv("train.csv")

    data_train = data_clear(data_train)
    test_df = data_clear(data_test)

    corr_matrix = data_train.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.show()

    train, validate = np.split(data_train.sample(frac=1), [int(.8 * len(data_train))])

    train_y = pd.get_dummies(train['playlist_genre'])
    train_x = train.drop(columns='playlist_genre')
    val_y = pd.get_dummies(validate['playlist_genre'])
    val_x = validate.drop(columns='playlist_genre')
    test_y = test_df['playlist_genre']
    test_x = test_df.drop(columns='playlist_genre')

    model = kr.Sequential()
    model.add(kr.layers.Dense(150, input_dim=12, activation="sigmoid"))
    model.add(kr.layers.Dense(50, activation="sigmoid"))
    model.add(kr.layers.Dense(6, activation="sigmoid"))

    model.summary()
    optimizer = kr.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    early_stop = kr.callbacks.EarlyStopping(monitor='val_loss', patience=50)
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
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    print("Done.")
