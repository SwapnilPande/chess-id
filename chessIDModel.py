from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten

# num_classes is the number of output classes
def getModel(numClasses, inputDim):
    # Create base pre-trained model
    baseModel = Xception(weights='imagenet', include_top=False, input_shape = inputDim)

    # Get output of xception
    xceptionOutput = baseModel.output
    xceptionDropout = Dropout(0.5)(xceptionOutput)
    xceptionFlat = Flatten()(xceptionDropout)

    # Layers appended to xception to train
    chessIDDense = Dense(1024, activation='relu')(xceptionFlat)
    chessIDDropout = Dropout(0.5)(chessIDDense)
    chessIDOutput = Dense(numClasses, activation='softmax')(chessIDDropout)

    # Create model object
    model = Model(inputs = baseModel.input, outputs = chessIDOutput)

    # Freeze all of the xception layers
    for layer in baseModel.layers:
        layer.trainable = False

    return model


