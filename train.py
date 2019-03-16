import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import shutil, os
import argparse


import chessIDModel

#Retrieve command line options
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Path to model checkpoint to import weights from")
parser.add_argument("-a", "--train-all-layers", help="Make all layers of the neural network trainable", action='store_true')
parse.add_argument("-n", "--num-layer-to-train", help="Specify number of layers of xception model to train, starts from end", default = 0)
parse.add_argument("-lr", "--learning-rate", help="Learning rate to use to for adam optimizer", default=0.0001)
args =  parser.parse_args()


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
adam = Adam(lr = args.lr)

if(args.m is not None):
    # Load new model - training of xception layers is disabled by default
    model = chessIDModel.getModel(len(categories), (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH, 3))
else:
    # Else we load the model
    model = load_model(args.m)

model.compile(adam, loss= "categorical_crossentropy", metrics = ["accuracy"])
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


