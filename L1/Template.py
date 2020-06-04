# Excercise 1: Regression
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
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from matplotlib.animation import FuncAnimation

# The syntax with the "()" is used to denote that this is a subclass
# of the nn.Module class, which is implemented by the PyTorch developers.
# It implements all of the utilitarian functionality of a machine 
# learning model so that we can focus on the actual implementation.
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()

		# Define and initialize neural network layers here.
		# This looks something like:
		# self.layer1 = nn.Linear(2, 1)
		# self.activation_fn = nn.Tanh()

	def forward(self, x):
		# Evaluate the layers in order and apply any
		# activation functions that you see fit. This looks like:
		# x = self.activation_fn(self.layer1(x))
		# x = self.activation_fn(self.layer2(x))

		return x

if __name__ == '__main__':
	# Change the number here to alter the number of CPU cores
	# that PyTorch will use. If you're computer isn't very powerful,
	# then you may need to keep this at one, to avoid freezing everything up.
	torch.set_num_threads(1)
	
	# This code generates datapoints for the function you are trying to 
	# replicate, as well as plotting it for you.

	datawidth = 128
	rng       = 0.18

	x = np.linspace(-rng, rng, datawidth)
	y = np.linspace(-rng, rng, datawidth)

	X, Y = np.meshgrid(x, y)

	theta = np.arctan2(Y, X)
	r     = np.sqrt(np.square(X) + np.square(Y))

	g1 = np.exp(-np.square(r))
	s1 = np.square(np.sin(3*theta - 6*r))

	# The weird power of 4 factors in the last part ar meant 
	# to keep the function well behaved at y ~ x ~ 0. Without
	# this factor, the function is poorly defined around the origin.
	Z  = s1 * g1 * ((1e4*np.power(r, 4)) / (1e4*np.power(r, 4) + 1))

	plt.imshow(
		Z, cmap='Blues_r', 
		interpolation='Bilinear', 
		extent=[-rng, rng, -rng, rng]
	)
	plt.colorbar()
	plt.title("Sample Function")
	plt.show()


	# This is where you split the data into a training dataset
	# and a test dataset. You need to use a small subset of the
	# amount of available data. Generally, the data necessary to
	# plot the function is way more than what is necessary to
	# learn it.

	# Here we are creating a sequential list of indices corresponding
	# to the datapoints. We then use the numpy random choice function
	# to select a subset of indices. This is necessary, because we need
	# to use that set of indices to select from multiple arrays. Otherwise,
	# we would just directly pass the array to the numpy random choice 
	# function and skip the part where we make a list of indices.
	indices = np.arange(datawidth**2)
	ratio   = 0.12
	mainset = np.random.choice(
		indices, int(ratio * datawidth**2), replace=False
	)

	# Here we reshape the inputs into the correct format for PyTorch.
	# PyTorch expects its input arrays to look something like this:
	# array([
	#     [x1, y1],
	#     [x2, y2],
	#     ...,
	#     [xN, yN]
	# ])
	# For this specific example, the neural network takes two inputs. 
	flattened_inputs = np.concatenate((
		X.reshape(datawidth**2),
		Y.reshape(datawidth**2)
	)).reshape(2, datawidth**2).T

	flattened_outputs = Z.reshape(datawidth**2)

	# This passes an array of indices to the numpy function that
	# handles selection of array elements. This is the part that
	# actually splits the dataset up.
	selected_inputs  = flattened_inputs[mainset]
	selected_outputs = flattened_outputs[mainset]

	# Now that we've extracted our dataset we can split it.
	# We now perform the same kind of process, except that
	# we are splitting into training and validation data,
	# rather than just deciding what part of the dataset to use.
	indices = np.arange(int(ratio * datawidth**2)) 
	ratio   = 0.7
	
	training_indices = np.random.choice(
		indices, int(ratio * len(indices)), replace=False
	)
	validation_indices = [i for i in indices if i not in training_indices]

	# Print some info about the training set. We want to make sure that we don't
	# have a model with twice as many parameters as there are data points. That
	# would likely lead to either severe overfit or severe underfit.
	print("Total Data Points      = %d"%(datawidth**2))
	print("Training Data Points   = %d"%len(indices))
	print("Validation Data Points = %d"%len(validation_indices))

	# Here we use the new set of indices to split the training and validation 
	# sets up. The .reshape call is necessary to get the data into a format
	# that PyTorch expects. Example:
	# array([
	#     [z1],
	#     [z2],
	#     ...,
	#     [zN]
	# ])
	# This makes more sense when you have a neural network with more than
	# one output. In this case, it just seems unecessary.
	training_in  = selected_inputs[training_indices]
	training_out = selected_outputs[training_indices].reshape(len(training_indices), 1)

	validation_in  = selected_inputs[validation_indices]
	validation_out = selected_outputs[validation_indices].reshape(len(validation_indices), 1)

	# This checks your system to see uf cuda is installed and properly configured.
	# If you run into issues, replace this whole block with "device = 'cpu'"
	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'

	# Here we convert these to PyTorch tensors and send them to the GPU
	# if there is one available.

	# These lines will simultanesouly convert the numpy arrays into Torch tensors,
	# convert them into 32-bit floating points (standard for ML) and move them
	# to the GPU if there is one.
	training_in    = torch.tensor(training_in).type(torch.FloatTensor).to(device)
	training_out   = torch.tensor(training_out).type(torch.FloatTensor).to(device)
	validation_in  = torch.tensor(validation_in).type(torch.FloatTensor).to(device)
	validation_out = torch.tensor(validation_out).type(torch.FloatTensor).to(device)

	# Now we initialize the objects necessary to train the neural network
	# as well as the network itself. Here, you will need to create an
	# optimizer, loss function and optionally, a learning rate scheduler.

	network = NeuralNetwork().to(device)

	# This calculates and prints the total number of parameters in the neural 
	# network.
	n_params  = sum([p.numel() for p in network.parameters()])
	print("Number of Parameters   = %d"%n_params)

	# These are all fairly standard parts of a neural network training process.
	# This is whats called "epoch training", where you train the network on the
	# entire training set all at once. An alternative is called "minibatch training",
	# where you split the training set into chunks and alternate between them when
	# training.
	max_iterations   = 30000
	val_interval     = 100
	record_interval  = 10
	training_error   = []
	validation_error = []
	records          = []

	

	# This is part of the code used to record snapshots. Its just a flattened
	# version of the x and y coordinates at which the network needs to be evaluated
	# to make a snapshot.
	test_data = torch.tensor(flattened_inputs).type(torch.FloatTensor).to(device)


	# This error catching block allows you to terminate early by
	# hitting Ctrl-C
	try:
		for i in range(max_iterations):
			# Perform the primary training process here.
			# Also add the training error to the array so it
			# can be plotted. 


			if i % val_interval == 0:
				with torch.no_grad():
					# This is where you check your validation error. 
					# You also want to add it to an array so you can plot
					# it later.

					


				# Report the current error to the user.
				print("%06d / %06d t: %2.4f v: %2.4f"%(i, max_iterations, terr, verr))

			# This will record a snapshot of the neural network output so
			# that the training process can be visualized later. Since the network
			# trains more slowly at higher iterations, this code will make the wait
			# period between snapshots increase with each snapshot. Otherwise the
			# animation will appear to slow to a stop when the network gets further
			# into its training process.
			if i % record_interval == 0:
				with torch.no_grad():
					z         = network(test_data).cpu()
					Z_net     = z.numpy().reshape(datawidth, datawidth)
					records.append(Z_net)
					record_interval *= 1.2
					record_interval  = int(record_interval)
					record_interval  = min(record_interval, 1000)

	except KeyboardInterrupt:
		print("\nQuitting Training . . . ")


	# The remaining code will plot the original function alongside an
	# animation of the output of the neural network. You should definitely
	# experiment with it when you have time.

	validation_x_axis = [val_interval*i for i in range(len(validation_error))]

	train, = plt.plot(range(len(training_error)), training_error)
	val,   = plt.plot(validation_x_axis, validation_error)
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.title("Loss vs. Training Iteration")
	plt.legend([train, val], ["Training", "Validation"])
	plt.show()

	# Now we eval the network at the same data points as the original graph
	# so that we can compare.

	network = network.to('cpu')
	network.eval() # Configure the network for use, instead of training

	with torch.no_grad():
		test_data = torch.tensor(flattened_inputs).type(torch.FloatTensor)
		z         = network(test_data)
		Z_net     = z.numpy().reshape(datawidth, datawidth).T


	fig, (ax1, ax2) = plt.subplots(1, 2)

	im1 = ax1.imshow(
		Z, cmap='Blues_r', 
		interpolation='Bilinear', 
		extent=[-rng, rng, -rng, rng]
	)
	ax1.set_title("Sample Function")

	im2 = ax2.imshow(
		Z_net.T, cmap='Blues_r',
		interpolation='Bilinear',
		extent=[-rng, rng, -rng, rng],
		vmin=Z.min(), vmax=Z.max()
	)
	ax2.set_title("Neural Network Function")

	def init():
		return im2, 

	def update(frame):
		im2.set_data(records[frame % len(records)])
		return im2, 

	ani = FuncAnimation(
		fig, 
		update, 
		frames=np.arange(len(records)),
        init_func=init, 
        blit=True
    )

	plt.show()