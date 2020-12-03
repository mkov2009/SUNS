import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


if __name__ == '__main__':

    df = pd.read_csv("data/styles.csv")
    df["id"] = df.apply(lambda row: str(row["id"]) + ".jpg", axis=1)

    batch_size = 32
    img_height = 32
    img_width = 32

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=0.2
    )

    train_ds = image_generator.flow_from_dataframe(
        dataframe=df,
        directory="data/images",
        x_col="id",
        y_col="gender",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset="training"
    )

    val_ds = image_generator.flow_from_dataframe(
        dataframe=df,
        directory="data/images",
        x_col="id",
        y_col="gender",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset="validation"
    )

    num_classes = 5

    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    epochs = 5
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
