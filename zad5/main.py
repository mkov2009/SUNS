import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
import numpy
import seaborn as sns


def create_model(i_height, i_width, n_classes):
    model_seq = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(i_height, i_width, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model_seq.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model_seq


def train(model, train_ds, val_ds, epochs, checkpoint_path):
    log_dir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    training = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback, cp_callback])

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


def load_weights(path, model):
    model.load_weights(path)
    return model


def predict(model, data):
    predicted = model.predict(data)

    predicted_classes = numpy.argmax(predicted, axis=1)

    true_classes = data.classes
    class_labels = list(data.class_indices.keys())

    cm = tf.math.confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


if __name__ == '__main__':

    df = pd.read_csv("data/styles.csv", error_bad_lines=False)
    df_own = pd.read_csv("data/own-styles.csv")

    df["id"] = df.apply(lambda row: str(row["id"]) + ".jpg", axis=1)
    df_own["id"] = df_own.apply(lambda row: str(row["id"]) + ".jpg", axis=1)

    batch_size = 50
    img_height = 32
    img_width = 32
    epochs = 5
    img_dir = "C:/images"
    img_test_dir = "data/own-images"

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

    test_ds = image_generator.flow_from_dataframe(
        dataframe=df_own,
        directory=img_test_dir,
        x_col="id",
        y_col="gender",
        target_size=(img_height, img_width),
        batch_size=batch_size,
    )

    num_classes = 5
    checkpoint_path = "saved/cp.ckpt"

    model = create_model(img_height, img_width, num_classes)

    train(model, train_ds, val_ds, epochs, checkpoint_path)
    # model = load_weights(checkpoint_path, model)
    # predict(model, test_ds)

