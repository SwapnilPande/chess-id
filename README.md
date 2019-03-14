# Chess ID
Forked repository to identify chessboard pieces for Chessbot project. Article about original repo: https://medium.com/@daylenyang/building-chess-id-99afa57326cd

The purpose of this project is to return the location of all pieces on the chessbot chessboard given a top-down image of the chessbot chessboard. The location of the board will be determined using two OpenCV ArUco markers located at opposite corners of the board. Based on the lcoation and orientation of these markers, each square will be segmented into separate images and inputted to a trained neural network to identify the piece.

## Changes from original repository:

* Modifying the original board detection algorithm to be more effective. However, it will only work for a chessboard with AruCo markers at the corner for identification (does not generalize to all chessboards like original repository)

* Porting chesspiece classification neural network to Keras/Tensorflow

* Replacing trained model used for transfer learning from AlexNet to Xception

* Building python package to predict chessboard piece positions to import into other python projects

* Modularizing the package

* Using pipenv as python package manager


## Dataset from original repository

The dataset used to train the model is available at the following link. This dataset was made available by the original author of this repository. https://www.dropbox.com/s/618l4ddoykotmru/Chess%20ID%20Public%20Data.zip?dl=0
