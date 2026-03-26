import cv2
import numpy as np
import os

def load_data(path):
    labels = {
        'Normal': 0,
        'Tuberculosis': 1
    }

    X, y = [], []

    for cls in os.listdir(path):
        for file in os.listdir(f"{path}/{cls}"):
            img = cv2.imread(f"{path}/{cls}/{file}")
            img = cv2.resize(img, (120, 120))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(120, 120, 1)
            img = img / 255.0

            X.append(img)
            y.append(labels[cls])

    return np.array(X), np.array(y)
