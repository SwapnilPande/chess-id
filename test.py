import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,load_model
import shutil, os

modelPath = "./checkpoints/chess-id-checkpoint-15-5.52.hdf5"

# Image dimension
SQUARE_SIDE_LENGTH = 227

# Size of training dataset
TEST_DATASET_SIZE = 185

# Relative directory to data
testDir = "./processed_data/output_test/"

# Categories for neural network to predict
categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

# Create dataset generator
batchSize = 32
testDataGenerator = ImageDataGenerator(rotation_range=270, horizontal_flip=True, rescale=1./255).flow_from_directory(
    directory = testDir,
    target_size = (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH),
    classes = categories,
)
# Load Model
model = load_model(modelPath)
loss = model.evaluate_generator(testDataGenerator,
            steps = TEST_DATASET_SIZE // batchSize,
                        workers = 10
)
print(loss)
