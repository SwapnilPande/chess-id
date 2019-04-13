import os
import cv2

# Path to directory containing images of chessboards
dataInputDir = ""

# Path to file containing coordinates for chessboard squares
coordinatePath = "square_locations.txt"

# Directory to store segmented and labelled images
dataOutputDir = ""

# Ingest square coordinates from file
squareCoords = []
with open(coordinatePath) as coordFile:
    for line in coordFile:
        line = line.replace("\n", "")
        squareCoords.append([int(x) for x in line.split(",")])


boardImages = os.listdir(dataInputDir)

for image in boardImages:
    # Get filename without extension
    filename, _ = os.splittext(image)

    # Read in image
    frame = cv2.imread(filename)

    # Iterate over all square coordinates and segment image
    for i, squareCoord in enumerate(squareCoords):
        x1 = squareCoord[0]
        y1 = squareCoord[1]
        x2 = squareCoord[2]
        x2 = squareCoord[3]

        # Slice image
        square = frame[y1:y2,x1:x2,:]

        cv2.imshow(str(i), square)
        label = raw_input("What chess piece is this?\n")

        saveDir = os.path.join(dataOutputDir, label)

        # Save image in the correct directory
        cv2.imwrite(filename + "-" + str(i), label)


