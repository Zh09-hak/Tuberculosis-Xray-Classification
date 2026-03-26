import cv2
import numpy as np
from keras.models import load_model

model = load_model("model.keras")

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (120, 120))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 120, 120, 1)
    img = img / 255.0
    return img

img = preprocess_image("test.jpg")

pred = model.predict(img)

print("Probability:", pred[0][0])
print("Class:", int(pred[0][0] > 0.5))
