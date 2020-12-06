import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf


if __name__ == '__main__':

    df = pd.read_csv("data/styles.csv")
    df["id"] = df.apply(lambda row: str(row["id"]) + ".jpg", axis=1)

    batch_size = 50
    img_height = 60
    img_width = 60
    epochs = 10
    img_dir = "C:/images"

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255.,
        validation_split=0.2
    )

    train_ds = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=img_dir,
        x_col="id",
        y_col="gender",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset="training"
    )

    val_ds = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=img_dir,
        x_col="id",
        y_col="gender",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset="validation"
    )

    num_classes = 5

    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(60, 60, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    print(model.summary())

    log_dir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    training = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback])

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
