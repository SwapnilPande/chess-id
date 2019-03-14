from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

# num_classes is the number of output classes
def get_model(numClasses):
    # Create base pre-trained model
    baseModel = Xception(weights='imagent', include_top=False)

    # Get output of xception
    xceptionOutput = baseModel.xceptionOutput
    xceptionDropout = Dropout(0.5)(xceptionOutput)

    # Layers appended to xception to train
    chessIDDense = Dense(1024, activation='relu')(xceptionDropout)
    chessIDDropout = Dropout(0.5)(chessIDDense)
    chessIDOutput = Dense(numClasses, activation='softmax')(chessIDDropout)

    # Create model object
    model = Model(inputs = baseModel.input, outputs = chessIDOutput)

    # Freeze all of the xception layers
    for layer in baseModel.layers:
        layer.trainable = False

    return model


