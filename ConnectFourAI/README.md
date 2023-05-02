# Connect Four AI

This repository contains the implementation of an AI for the game of Connect Four, where the objective is to be the first player to form a horizontal, vertical, or diagonal line of four of one's own discs. The AI leverages machine learning techniques to make intelligent decisions while playing the game.

## Game Rules

Connect Four is a two-player connection board game, in which players choose a color and then take turns dropping colored discs into a seven-column, six-row vertically suspended grid. The pieces fall straight down, occupying the lowest available space within the column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs.

## Data

The dataset used for training the AI is sourced from John Tromp (tromp '@' cwi.nl). It contains all legal 8-ply positions in the game of Connect Four in which neither player has won yet, and in which the next move is not forced. Player "x" is the first player, and player "o" is the second. The outcome class represents the game theoretical value for the first player.

## Model

Our AI is based on an MLP (Multilayer Perceptron) model, which is saved as `9.pkl`. The MLP model is a type of feedforward artificial neural network that is capable of learning patterns and making predictions based on the input data.

## Usage

You can try playing against the AI by running the provided Jupyter Notebook (`.ipynb` file). The notebook contains step-by-step instructions on how to play the game and interact with the AI.

