import code
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# The classes in this file are all have a hard coded number of layers
# because the introspection code that pytorch uses to retrieve a list
# of parameters requires it. There is a workaround, but I don't feel 
# like writing it yet.

class Net5(nn.Module):
	def __init__(self, n_in, n_out, dropout_p=0.0):
		super(Net5, self).__init__()

		self.insize  = n_in
		self.outsize = n_out

		self.drop = dropout_p
		if self.drop != 0.0:
			self.dropout_layer = nn.Dropout(p=self.drop)

		self.conv1 = nn.Conv2d(1, 4, 6, padding=3)
		self.pool  = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(4, 4, 6, padding=3)

		N, C_out, H_out, W_out = self.conv_2d_output_shape(
			1, 1, self.insize, self.insize, self.conv1
		)
		N, C_out, H_out, W_out = self.conv_2d_output_shape(
			1, 4, int(H_out / 2), int(W_out / 2), self.conv2
		)
		self.linear_one_input  = C_out * int(H_out / 2) * int(W_out / 2)
		self.linear_one_output = int(self.linear_one_input / 2)

		self.l1    = nn.Linear(self.linear_one_input, self.linear_one_output)
		self.l2    = nn.Linear(self.linear_one_output, 30)
		self.l3    = nn.Linear(30, self.outsize)

		

		
	# Returns the shape of the output of a convolutional layer 
	# based on its input. This assumes square strides and padding.
	def conv_2d_output_shape(self, N, C, H, W, conv):
		C_out = conv.out_channels
		H_out = (H + 2*conv.padding[0] - conv.dilation[0]*(conv.kernel_size[0] - 1) - 1)
		H_out = (H_out / conv.stride[0]) + 1
		H_out = int(np.floor(H_out))

		W_out = (W + 2*conv.padding[1] - conv.dilation[1]*(conv.kernel_size[1] - 1) - 1)
		W_out = (W_out / conv.stride[1]) + 1
		W_out = int(np.floor(W_out))

		return N, C_out, H_out, W_out



	# Set train=False in order to turn off the dropout layer.
	def forward(self, x, train=True):
		if self.drop != 0.0 and train:
			x = self.dropout_layer(x)


		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, self.linear_one_input)
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)

		return x