import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import shutil, os

import chessIDModel

# Image dimension
SQUARE_SIDE_LENGTH = 227

# Size of training dataset
TRAIN_DATASET_SIZE = 10360

# Relative directories to data
trainDir = "./Chess ID Public Data/output_train/"
testDir = "./processed_data/output_test/"


# Categories for neural network to learn
categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

# Create dataset generator
batchSize = 32
trainDataGenerator = ImageDataGenerator(rotation_range=270, horizontal_flip=True, rescale=1./255).flow_from_directory(
    directory = trainDir,
    target_size = (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH),
    classes = categories,
    batch_size = batchSize,
    shuffle = True,

)

# Define checkpoint callback
shutil.rmtree("./checkpoints", True)
os.mkdir("./checkpoints")
callbacks = [ModelCheckpoint('./checkpoints/chess-id-checkpoint-{epoch:02d}-{loss:.2f}.hdf5')]

# Instantiate model
#sgd = SGD(lr = 0.01, momentum = 0.9)
model = chessIDModel.getModel(len(categories), (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH, 3))
model.compile("adam", loss= "categorical_crossentropy", metrics = ["accuracy"])
model.summary()
model.fit_generator(
			trainDataGenerator,
            steps_per_epoch = TRAIN_DATASET_SIZE // batchSize,
			epochs = 50,
			workers = 10,
			shuffle=True,
            callbacks = callbacks
		)

# Save final model
model.save("./checkpoints/finalModel.hdf5")


