import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_image(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    return img


def load_data():
    df = pd.read_csv("../dataset/driving_log.csv", header=None)

    data_list = []
    label_list = []

    for _, row in df.iterrows():
        center = row[0].strip()
        angle = float(row[3])

        img = load_and_preprocess(center)
        data_list.append(img)
        label_list.append(angle)

    x = np.array(data_list)
    y = np.array(label_list)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    return x_train, x_test, y_train, y_test