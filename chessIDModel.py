from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten

# num_classes is the number of output classes
def getModel(numClasses, inputDim, trainAllLayers, numLayerToTrain):
    # Create base pre-trained model
    baseModel = Xception(weights='imagenet', include_top=False, input_shape = inputDim)

    # Get output of xception
    xceptionOutput = baseModel.output
    #xceptionDropout = Dropout(0.5)(xceptionOutput)
    xceptionFlat = Flatten()(xceptionOutput)

    # Layers appended to xception to train
    chessIDDense = Dense(10, activation='relu')(xceptionFlat)
    #chessIDDropout = Dropout(0.5)(chessIDDense)
    chessIDOutput = Dense(numClasses, activation='softmax')(chessIDDense)

    # Create model object
    model = Model(inputs = baseModel.input, outputs = chessIDOutput)

    print(len(baseModel))

    # Freeze all of the xception layers
    if(not trainAllLayers):
        # Disable training of all layers in xception model
        for layer in baseModel.layers[]:
            layer.trainable = False

        #Enable last n layers to train
        for layer in baseModel.layers[-1*numLayerToTrain:]:
            layer.trainable = True
    # else, we train all layers in the network, including xception layers

    return model

