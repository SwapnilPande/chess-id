import os
import cv2

# Path to directory containing images of chessboards
dataInputDir = "./Board1/"

# Path to file containing coordinates for chessboard squares
coordinatePath = "square_locations.txt"

# Directory to store segmented and labelled images
dataOutputDir = "./output"

# Ingest square coordinates from file
squareCoords = []
with open(coordinatePath) as coordFile:
    for line in coordFile:
        line = line.replace("\n", "")
        squareCoords.append([int(x) for x in line.split(",")])


boardImages = os.listdir(dataInputDir)

print(boardImages)
for image in boardImages:
    print("Current images: " + image)
    # Get filename without extension
    filename, _ = os.path.splitext(image)

    # Read in image
    frame = cv2.imread(os.path.join(dataInputDir, image))

    # Iterate over all square coordinates and segment image
    for i, squareCoord in enumerate(squareCoords):
        x1 = squareCoord[0]
        y1 = squareCoord[1]
        x2 = squareCoord[2]
        y2 = squareCoord[3]

        print(squareCoord)

        # Slice image
        square = frame[y1:y2,x1:x2,:]

        square = cv2.resize(square, (227,227))

        cv2.imshow(str(i), square)
        cv2.waitKey(1000)
        label = input("What chess piece is this?\n")


        saveDir = os.path.join(dataOutputDir, label)

        # Save image in the correct directory
        outputPath = os.path.join(dataOutputDir, label, filename + "-" + str(i) + ".png")
        cv2.imwrite(outputPath, square)


        cv2.destroyAllWindows()



