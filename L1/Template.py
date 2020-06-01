# Goals of this exercise:
# 1) Learn how to convert raw data into some kind of training data
# 2) Learn how to properly perform a training-validation split
# 3) Learn how to perform a k-fold cross validation
# 4) Experiment with activation functions
# 5) Experiment with different optimization algorithms
# 6) Demonstrate the fact that neural networks extrapolate poorly

import time
import os
import sys
import code

import matplotlib.pyplot as plt
import numpy             as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()

		# Define and initialize neural network layers here.

	def forward(self, x):
		# Evaluate the layers in order and apply any
		# activation functions that you see fit.

		return x

if __name__ == '__main__':
	
	# This is the function we are trying to emulate.

	x = np.linspace(-1, 1, 512)
	y = np.linspace(-1, 1, 512)

	X, Y = np.meshgrid(x, y)

	theta = np.arctan2(Y, X)
	r     = np.sqrt(np.square(X) + np.square(Y))

	g1 = np.exp(-np.square(r))
	s1 = np.square(np.sin(3*theta - 6*r))

	Z  = s1 * g1 * ((1e4*np.power(r, 4)) / (1e4*np.power(r, 4) + 1))

	plt.imshow(
		Z, cmap='Blues_r', 
		interpolation='Bilinear', 
		extent=[-1, 1, -1, 1]
	)
	plt.colorbar()
	plt.title("Sample Function")
	plt.show()