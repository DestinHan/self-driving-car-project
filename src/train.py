import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import show_result

from sklearn.model_selection import train_test_split

from model import build_model
from preprocess import preprocess_image, balance_data
from augmentation import augment_image


def batch_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_images = []
        batch_steering = []

        for _ in range(batch_size):
            index = random.randint(0, len(image_paths) - 1)

            image_path = image_paths[index]
            steering = steering_angles[index]

            img = cv2.imread(image_path)

            if img is None:
                continue

            if is_training:
                img, steering = augment_image(img, steering, training_model=True)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_image(img)

            batch_images.append(img)
            batch_steering.append(steering)

        yield np.array(batch_images), np.array(batch_steering)


def main():
    print("Loading driving log...")

    data = pd.read_csv(
        "dataset/driving_log.csv",
        header=None,
        names=["center", "left", "right", "steering", "throttle", "brake", "speed"]
    )

    data = balance_data(data)

    print(data.head())
    print(data.columns)

    image_paths = data["center"].values
    steering_angles = data["steering"].astype(float).values

    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths,
        steering_angles,
        test_size=0.2,
        random_state=42
    )

    print("Train samples:", len(X_train))
    print("Validation samples:", len(X_valid))

    model = build_model()
    model.summary()

    batch_size = 64
    epochs = 20

    train_generator = batch_generator(X_train, y_train, batch_size, True)
    valid_generator = batch_generator(X_valid, y_valid, batch_size, False)

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(X_train) // batch_size),
        validation_data=valid_generator,
        validation_steps=max(1, len(X_valid) // batch_size),
        epochs=epochs,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    print("Model saved to models/model.h5")

    os.makedirs("outputs", exist_ok=True)

    show_result(history, epochs)
    print("Loss plot saved to outputs/loss_plot.png")


if __name__ == "__main__":
    main()