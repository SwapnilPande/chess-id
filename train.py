import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
import shutil, os
import argparse
import math
from models import vgg_with_dense_128

# Set allow growth to train on RTX 2060 card
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


import chessIDModel

#Retrieve command line options
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Path to model checkpoint to import weights from")
parser.add_argument("-a", "--train-all-layers", help="Make all layers of the neural network trainable", action='store_true')
parser.add_argument("-n", "--num-layers-to-train", help="Specify number of layers of xception model to train, starts from end", default = 0)
parser.add_argument("-lr", "--learning-rate", help="Learning rate to use to for adam optimizer", default=0.0001)
args =  parser.parse_args()


# Image dimension
SQUARE_SIDE_LENGTH = 227

# Size of training dataset
TRAIN_DATASET_SIZE = 41440
TEST_DATASET_SIZE = 740

# Relative directories to data
#trainDir = "/data/Chess ID Public Data/output_train/"
trainDir = "/data/processed_data/output_train/"
testDir = "/data/processed_data/output_test/"


# Categories for neural network to learn
categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

# Create dataset generator
batchSize = 64
# rotation_range=270, horizontal_flip=True
trainDataGenerator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory = trainDir,
    target_size = (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH),
    classes = categories,
    batch_size = batchSize,
    shuffle = True)
    #horizontal_flip=True,
testDataGenerator = ImageDataGenerator( rescale=1./255).flow_from_directory(
    directory = testDir,
    target_size = (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH),
    classes = categories,
)

# Define checkpoint callback
shutil.rmtree("./checkpoints", True)
os.mkdir("./checkpoints")
callbacks = [ModelCheckpoint('./checkpoints/chess-id-checkpoint-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5')]

# Instantiate model
sgd = SGD(lr = float(args.learning_rate), momentum = 0.9)

if(args.model is None):
    # Load new model - training of xception layers is disabled by default
    model = chessIDModel.getModel()
else:
    # Else we load the model
    model = load_model(args.model)


# Set the correct number of trainable layers
model = chessIDModel.setTrainableLayers(model, args.train_all_layers, int(args.num_layers_to_train))
model.compile(sgd, loss = "categorical_crossentropy", metrics = ["accuracy"])

model.summary()
model.fit_generator(
			trainDataGenerator,
            validation_data = testDataGenerator,
            validation_steps = math.ceil(TEST_DATASET_SIZE/batchSize),
            steps_per_epoch = math.ceil(TRAIN_DATASET_SIZE/batchSize),
			epochs = 25,
			workers = 10,
			shuffle=True,
            callbacks = callbacks
		)
# Save final model
model.save("./checkpoints/finalModel.hdf5")


