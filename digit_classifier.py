import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/digit_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    resized = cv2.resize(thresh, (28, 28)).astype("float32") / 255.0
    return resized.reshape(1, 28, 28, 1)

def classify_digit(cell_img):
    processed = preprocess_cell(cell_img)
    prediction = model.predict(processed)
    return np.argmax(prediction)
