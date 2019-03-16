import numpy as np
import os, shutil, glob
import cv2
# import scipy.spatial as spatial
# import scipy.cluster as clstr
# from collections import defaultdict
# from sklearn.utils import shuffle
# import os, glob, skimage, cv2, shutil

# Relative directories to data
TRAIN_DIRS = {'input': './Chess ID Public Data/output_train/', 'output': './processed_data/output_train/'}
TEST_DIRS = {'input': './Chess ID Public Data/output_test/',  'output': './processed_data/output_test/'}

# Image dimension
SQUARE_SIDE_LENGTH = 227

# Generate a 2D rotation matrix
M_90deg_rotation = cv2.getRotationMatrix2D((SQUARE_SIDE_LENGTH / 2, SQUARE_SIDE_LENGTH / 2), 90, 1)

# Generate transforms for a single input image
def writeAugmentedOutput(outputDir, imagePath, label):
    # Retrieve name of image without extension
    _, imageNameWithExtension = os.path.split(imagePath)
    imageName, _  = os.path.splitext(imageNameWithExtension)

    image = cv2.imread(imagePath)

    # Generate transformed images
    image90 = cv2.warpAffine(image, M_90deg_rotation, (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH))
    image180 = cv2.warpAffine(image90, M_90deg_rotation, (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH))
    image270 = cv2.warpAffine(image180, M_90deg_rotation, (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH))

    cv2.imwrite(os.path.join(outputDir, label, imageName + ".jpg"), image)
    cv2.imwrite(os.path.join(outputDir, label, imageName + "_90.jpg"), image90)
    cv2.imwrite(os.path.join(outputDir, label, imageName + "_180.jpg"), image180)
    cv2.imwrite(os.path.join(outputDir, label, imageName + "_270.jpg"), image270)

# Generates augmented data for all images
def generateAugmentedData(dir_dict):
    # Ensure path names are formatted correctly
    for path in dir_dict.values():
        assert path[-1] == '/'

    # Create output directories
    shutil.rmtree(dir_dict['output'], True)
    os.makedirs(dir_dict['output'])

    # Subdirectories for each label within input directory
    labelDirs = os.listdir(dir_dict['input'])
    # Create subdirectory for each label in output directory
    for label in labelDirs:
        os.makedirs(os.path.join(dir_dict['output'], label))

    inputPaths = glob.glob(dir_dict['input'] + '*/*.jpg', recursive=True)
    percent = 0
    for i in range(len(inputPaths)):
        # Print completion percentage
        new_percent = 100 * float(i) / len(inputPaths)
        if new_percent > percent + 5:
            percent = new_percent
            print(str(percent) + '%')

        # Retrieve input path
        imagePath = inputPaths[i]

        # Retrieve image label
        labelDir, _ = os.path.split(imagePath)
        _ , imageLabel = os.path.split(labelDir)

        # Generate augmented images and write them to output directory
        writeAugmentedOutput(dir_dict['output'], imagePath, imageLabel)
    print('Done')

if(__name__ == "__main__"):
    print("Generating Training Data")
    generateAugmentedData(TRAIN_DIRS)
    print("Generating Testing Data")
    generateAugmentedData(TEST_DIRS)


