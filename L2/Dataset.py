import json
import cv2
import os
import code
import sys
import torch

import numpy as np

from Progress import ProgressBar

class HandwrittenDataset:
	def __init__(self, path, max_load=None):
		self.path = path

		with open(os.path.join(self.path, 'meta.json'), 'r') as file:
			self.meta = json.loads(file.read())['classes']

		self.data = []
		
		files = [e for e in os.listdir(self.path) if e.split('.')[-1] == 'npy']
		files = sorted(files)

		if max_load is not None:
			idx   = np.random.choice(range(len(files)), max_load, replace=False)
			f = []
			m = []
			for i in idx:
				f.append(files[i])
				m.append(self.meta[i])
			self.meta = m
			files = f

		pb = ProgressBar("Loading", 15, len(files), update_every=2000, ea=15)
		for i, file in enumerate(files):
			f = os.path.join(self.path, file)
			self.data.append(np.load(f))
			pb.update(i + 1)

		pb.finish()

	def configure(self, split=0.9, device='cpu'):
		charset  = "abcdefghijklmnopqrstuvwxyz"
		charset += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		charset += "0123456789"

		self.class_meta = {}
		for i, c in enumerate(charset):
			self.class_meta[c] = i


		self.n_train = int(round(split * len(self.data)))
		self.n_val   = len(self.data) - self.n_train

		self.train_indices = np.random.choice(
			len(self.data), self.n_train, replace=False
		)

		self.val_indices = [
			v for v in range(len(self.data)) if v not in self.train_indices
		]

		self.train_inputs = [self.data[i] for i in self.train_indices]
		self.val_inputs   = [self.data[i] for i in self.val_indices]

		self.train_labels = [
			self.class_meta[self.meta[i]] for i in self.train_indices
		]

		self.val_labels = [
			self.class_meta[self.meta[i]] for i in self.val_indices
		]

		self.t_train_inputs = torch.tensor(self.train_inputs)
		self.t_train_inputs = self.t_train_inputs.type(torch.FloatTensor)
		self.t_train_inputs = self.t_train_inputs.reshape(
			len(self.train_inputs), 1, self.data[0].shape[0], self.data[0].shape[1] 
		)
		self.t_train_inputs = self.t_train_inputs.to(device)

		self.t_train_labels = torch.tensor(self.train_labels)
		self.t_train_labels = self.t_train_labels.to(device)

		self.t_val_inputs = torch.tensor(self.val_inputs)
		self.t_val_inputs = self.t_val_inputs.type(torch.FloatTensor)
		self.t_val_inputs = self.t_val_inputs.reshape(
			len(self.val_inputs), 1, self.data[0].shape[0], self.data[0].shape[1] 
		)
		self.t_val_inputs = self.t_val_inputs.to(device)

		self.t_val_labels = torch.tensor(self.val_labels)
		self.t_val_labels = self.t_val_labels.to(device)


		return (
			self.t_train_inputs,
			self.t_train_labels,
			self.t_val_inputs,
			self.t_val_labels
		)

	def lookupTrainingInput(self, idx):
		index = self.train_indices[idx]
		return index, self.data[index], self.meta[index]

	def lookupValidationInput(self, idx):
		index = self.val_indices[idx]
		return index, self.data[index], self.meta[index]




if __name__ == '__main__':
	d = HandwrittenDataset('../nist_19_24/', max_load=50000)
	d.configure(device='cuda:0')
	code.interact(local=locals())