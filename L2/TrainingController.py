import code
import sys
import argparse
import os
import time
import json
import shutil
import cv2
import numpy             as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrsched

from Network             import Net5
from Progress            import ProgressBar


# This class is concerned with controlling the training process for
# the top level network (the one that classifies by group). It handles
# training, recording error, learning rate schedulers, optimization
# algorithms, etc.
class Controller:
	def __init__(self, training_set, network, seed=None, **kwargs):
		if seed is not None:
			torch.manual_seed(seed)
			np.random.seed(seed)

		self.processKwargs(kwargs)
		self.validateArgs()

		self.network = network

		self.opt       = self.initializeOptimizer()
		self.criterion = nn.CrossEntropyLoss()

		self.training_inputs   = training_set[0]
		self.training_labels   = training_set[1]
		self.validation_inputs = training_set[2]
		self.validation_labels = training_set[3]

		if self.sched:
			self.scheduler = lrsched.ReduceLROnPlateau(
				self.opt, mode='max', 
				threshold=0.0001, 
				patience=10, 
				verbose=self.verbose, 
				factor=0.75
			)


	def getClassificationAccuracy(self, actual, correct):
		n_correct = 0

		for row in range(actual.shape[0]):
			act = actual[row, :]
			cor = correct[row]

			if torch.argmax(act) == cor:
				n_correct += 1

		return n_correct / actual.shape[0]

	def train(self):
		training_acc       = []
		validation_acc     = []
		cross_entropy_loss = []
		timings            = []

		for idx in range(self.max_iter):
			try:
				self.opt.zero_grad()
				start  = time.time_ns()
				output = self.network(self.training_inputs)
				stop   = time.time_ns()
				timings.append(stop - start)
				loss   = self.criterion(output, self.training_labels)

				loss.backward()
				self.opt.step()
			except:
				print("")
				print("Terminating early . . . ")
				break

			cross_entropy_loss.append(loss.item())

			if idx % self.val_freq == 0:
				with torch.no_grad():
					output = self.network(self.training_inputs, train=False)
					t_acc = self.getClassificationAccuracy(
						output, 
						self.training_labels
					)

					output  = self.network(self.validation_inputs, train=False)
					v_acc = self.getClassificationAccuracy(
						output,
						self.validation_labels
					)

					training_acc.append(t_acc)
					validation_acc.append(v_acc)

				if self.verbose:
					print("iteration: %06d / %06d, "%(idx, self.max_iter), end='')
					print("cel: %1.4f, t_acc: %1.4f, v_acc: %1.4f"%(
						loss.item(), t_acc, v_acc
					))

				if self.sched:
					self.scheduler.step(v_acc)

		return (training_acc, validation_acc, cross_entropy_loss, timings)


	def initializeOptimizer(self):
		init = {
			'sgd'      : optim.SGD, 
			'adadelta' : optim.Adadelta, 
			'adagrad'  : optim.Adagrad, 
			'adam'     : optim.Adam,
			'adamw'    : optim.AdamW, 
			'adamax'   : optim.Adamax, 
			'asgd'     : optim.ASGD
		}

		if self.optimizer == 'sgd':
			opt = init[self.optimizer](
				self.network.parameters(),
				lr=self.lr,
				weight_decay=self.l2rs,
				momentum=self.momentum
			)
		else:
			opt = init[self.optimizer](
				self.network.parameters(),
				lr=self.lr,
				weight_decay=self.l2rs
			)

		return opt

	def validateArgs(self):
		valid_optimizers = [
			'sgd', 'adadelta', 'adagrad', 'adam',
			'adamw', 'adamax', 'asgd'
		]

		self.optimizer = self.optimizer.lower()
		if self.optimizer not in valid_optimizers:
			raise Exception("Unrecognized optimizer \'%s\'"%self.optimizer)

		if self.lr <= 0.0:
			raise Exception("lr must be >= 0.0")

		if self.momentum != 0.0 and self.optimizer != 'sgd':
			raise Exception("Momentum is only valid for \'sgd\' optimizer.")

		if self.l2rs < 0.0:
			raise Exception("l2rs must be >= 0.0")

		if self.max_iter < 1:
			raise Exception("max_iter must be >= 1")

		if self.val_freq < 1:
			raise Exception("val_freq must be >= 1")

	def processKwargs(self, kwargs):
		valid_keys = [
			'optimizer', 'lr',      'momentum', 
			'l2rs',      'sched',   'verbose',
			'max_iter',  'stop_fn', 'val_freq'
		]

		defaults = [
			'adamw', 0.01, 0.0,
			0.01,    True, True,
			10000,   None, 25
		]

		for k in kwargs:
			if k not in valid_keys:
				raise Exception("Invalid argument: %s"%k)


		if 'optimizer' not in kwargs:
			self.optimizer = defaults[valid_keys.index('optimizer')]
		else:
			self.optimizer = kwargs['optimizer']

		if 'lr' not in kwargs:
			self.lr = defaults[valid_keys.index('lr')]
		else:
			self.lr = kwargs['lr']

		if 'momentum' not in kwargs:
			self.momentum = defaults[valid_keys.index('momentum')]
		else:
			self.momentum = kwargs['momentum']

		if 'l2rs' not in kwargs:
			self.l2rs = defaults[valid_keys.index('l2rs')]
		else:
			self.l2rs = kwargs['l2rs']

		if 'verbose' not in kwargs:
			self.verbose = defaults[valid_keys.index('verbose')]
		else:
			self.verbose = kwargs['verbose']

		if 'sched' not in kwargs:
			self.sched = defaults[valid_keys.index('sched')]
		else:
			self.sched = kwargs['sched']

		if 'max_iter' not in kwargs:
			self.max_iter = defaults[valid_keys.index('max_iter')]
		else:
			self.max_iter = kwargs['max_iter']

		if 'stop_fn' not in kwargs:
			self.stop_fn = defaults[valid_keys.index('stop_fn')]
		else:
			self.stop_fn = kwargs['stop_fn']

		if 'val_freq' not in kwargs:
			self.val_freq = defaults[valid_keys.index('val_freq')]
		else:
			self.val_freq = kwargs['val_freq']