
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,load_model
import shutil, os

# Set allow growth to train on RTX 2060 card
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

modelPath = "./checkpoints/chess-id-checkpoint-24-0.03-0.99.hdf5"

# Image dimension
SQUARE_SIDE_LENGTH = 227

# Relative directory to data
imagePath = "/train/sample_images/Wrook1.png"

frame = cv2.imread(imagePath)

frame = cv2.resize(frame, (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH))

frame = np.expand_dims(frame, axis= 0)
print(frame.shape)

# Categories for neural network to predict
categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

# Load Model
model = load_model(modelPath)
out = model.predict(frame)
print(out)
