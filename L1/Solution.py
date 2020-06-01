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

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()

		# Define and initialize neural network layers here.
		self.l1   = nn.Linear(2, 64)
		self.l2   = nn.Linear(64, 32)
		self.l3   = nn.Linear(32, 16)
		self.l4   = nn.Linear(16, 1)
		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()

	def forward(self, x):
		# Evaluate the layers in order and apply any
		# activation functions that you see fit.

		x = self.tanh(self.l1(x))
		x = self.tanh(self.l2(x))
		x = self.tanh(self.l3(x))
		x = self.tanh(self.l4(x))

		return x

if __name__ == '__main__':
	torch.set_num_threads(1)
	
	# This is the function we are trying to emulate.

	datawidth = 128
	rng       = 0.18

	x = np.linspace(-rng, rng, datawidth)
	y = np.linspace(-rng, rng, datawidth)

	X, Y = np.meshgrid(x, y)

	theta = np.arctan2(Y, X)
	r     = np.sqrt(np.square(X) + np.square(Y))

	g1 = np.exp(-np.square(r))
	s1 = np.square(np.sin(3*theta - 6*r))

	Z  = s1 * g1 * ((1e4*np.power(r, 4)) / (1e4*np.power(r, 4) + 1))
	# Z = np.exp(-r**2)*np.square(np.sin(4*r))
	# Z = -np.exp(-X**2)*Y

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
	# amount of available data. 262,144 is way too many data points.

	indices = np.arange(datawidth**2)
	ratio   = 0.12
	mainset = np.random.choice(
		indices, int(ratio * datawidth**2), replace=False
	)

	flattened_inputs = np.concatenate((
		X.reshape(datawidth**2),
		Y.reshape(datawidth**2)
	)).reshape(2, datawidth**2).T

	flattened_outputs = Z.reshape(datawidth**2)

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

	print("Total Data Points      = %d"%(datawidth**2))
	print("Training Data Points   = %d"%len(indices))
	print("Validation Data Points = %d"%len(validation_indices))

	

	training_in  = selected_inputs[training_indices]
	training_out = selected_outputs[training_indices].reshape(len(training_indices), 1)

	validation_in  = selected_inputs[validation_indices]
	validation_out = selected_outputs[validation_indices].reshape(len(validation_indices), 1)

	# Here we convert these to PyTorch tensors and send them to the GPU
	# if there is one available.

	# if torch.cuda.is_available():
	# 	device = 'cuda:0'
	# else:
	# 	device = 'cpu'
	device = 'cpu'

	# These lines will simultanesouly conver the numpy arrays into Torch tensors,
	# convert them into 32-bit floating points (standard for ML) and move them
	# to the GPU if there is one.
	training_in    = torch.tensor(training_in).type(torch.FloatTensor).to(device)
	training_out   = torch.tensor(training_out).type(torch.FloatTensor).to(device)
	validation_in  = torch.tensor(validation_in).type(torch.FloatTensor).to(device)
	validation_out = torch.tensor(validation_out).type(torch.FloatTensor).to(device)

	# Now we initialize the objects necessary to train the neural network
	# as well as the network itself.
	learning_rate = 0.01
	momentum      = 0.9

	network   = NeuralNetwork().to(device)
	n_params  = sum([p.numel() for p in network.parameters()])
	print("Number of Parameters   = %d"%n_params)
	# optimizer = optim.SGD(
	# 	network.parameters(), 
	# 	lr=learning_rate, 
	# 	momentum=momentum
	# )
	optimizer = optim.AdamW(
		network.parameters(), 
		lr=learning_rate
	)
	criteria  = nn.L1Loss()
	scheduler = lrs.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5)

	max_iterations   = 30000
	val_interval     = 100
	record_interval  = 10
	training_error   = []
	validation_error = []
	records          = []

	z_reg = np.abs(Z).mean()
	test_data = torch.tensor(flattened_inputs).type(torch.FloatTensor).to(device)

	try:
		for i in range(max_iterations):
			optimizer.zero_grad()
			result = network(training_in)
			loss   = criteria(result, training_out)
			loss.backward()
			optimizer.step()
			terr = loss.item() / z_reg
			training_error.append(terr)
			scheduler.step(terr)

			if i % val_interval == 0:
				with torch.no_grad():
					result = network(validation_in)
					loss   = criteria(result, validation_out)
					verr   = loss.item() / z_reg
					validation_error.append(verr)

				print("%06d / %06d t: %2.4f v: %2.4f"%(i, max_iterations, terr, verr))

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

	validation_x_axis = [val_interval*i for i in range(len(validation_error))]

	train, = plt.plot(range(len(training_error)), training_error)
	val,   = plt.plot(validation_x_axis, validation_error)
	plt.xlabel("Iteration")
	plt.ylabel("MSE")
	plt.title("Error vs. Training Iteration")
	plt.legend([train, val], ["Training", "Validation"])
	plt.show()

	# Now we eval the network at the same data points as the original graph
	# so that we can compare.
	

	network = network.to('cpu')
	network.eval()

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