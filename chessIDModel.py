from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.regularizers import l2

def getModel():
    # Create base pre-trained model
    baseModel = VGG16(weights='imagenet', include_top=False, input_shape = (227, 227, 3))
    print(len(baseModel.layers))

    # Get output of xception
    vggOutput = baseModel.output
    vggDropout = Dropout(0.5)(vggOutput)
    vggFlat = Flatten()(vggDropout)

    ## Layers appended to xception to train
    chessIDDense = Dense(512,
        activation='relu',
        kernel_regularizer=l2(0.05)
    )(vggFlat)
    chessIDDropout = Dropout(0.5)(chessIDDense)
    chessIDOutput = Dense(13, activation='softmax')(chessIDDropout)

    # Create model object
    model = Model(inputs = baseModel.input, outputs = chessIDOutput)

    for layer in baseModel.layers:
        layer.trainable = False

    return model

def setTrainableLayers(model, trainAllLayers, numLayersToTrain):
    if(trainAllLayers):
        for layer in model.layers:
            layer.trainable = True
    else:
        for i in range(0, 19-numLayersToTrain):
            model.layers[i].trainable = False
        for i in range(19-numLayersToTrain, 19):
            model.layers[i].trainable = True
    return model


