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

from Dataset             import HandwrittenDataset
from Progress            import ProgressBar
from TrainingController  import Controller
from Network             import Net5

def preprocess():
	parser = argparse.ArgumentParser(
		description='Train a neural network on the specified dataset.'
	)

	parser.add_argument(
		'-d', '--dataset', dest='dataset', type=str, default='../test_set/',
		help='Path to the dataset images.'
	)

	parser.add_argument(
		'-q', '--save', dest='save_to', type=str, default='model.json',
		help='Where to save the trained model.'
	)

	parser.add_argument(
		'-c', '--cuda', dest='use_cuda', action='store_true',
		help='Use CUDA for training.'
	)

	parser.add_argument(
		'-s', '--split', dest='train_val_split', type=float, default=0.9,
		help='Ratio of training to total data.'
	)

	parser.add_argument(
		'-O', '--optimizer', dest='optimizer', type=str, default='adagrad',
		help='The optimizer to use. '+
		'(SGD, Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD)'
	)

	parser.add_argument(
		'-l2rs', '--l2-reg-strength', dest='l2_reg_strength', type=float, 
		default=0.0,
		help='L2 Regularization strength.'
	)

	parser.add_argument(
		'-lr', '--learning-rate', dest='learning_rate', type=float, 
		default=0.01,
		help='Learning Rate.'
	)

	parser.add_argument(
		'-m', '--momentum', dest='momentum', type=float, 
		default=0.0,
		help='Parameter valid for certain optimizers.'
	)

	parser.add_argument(
		'-r', '--reduce-on-plateau', dest='use_sched', action='store_true', 
		help='Use a reduce-on-plateau learning rate scheduler.'
	)

	parser.add_argument(
		'-D', '--dropout', dest='dropout_p', type=float, default=0.07, 
		help='Probability to use in dropout layer.'
	)

	parser.add_argument(
		'-S', '--seed', dest='seed', type=int, default=-1, 
		help='Seed to use for all randomization.'
	)

	parser.add_argument(
		'-i', '--iterations', dest='iterations', type=int, default=10000, 
		help='Number of training iterations.'
	)

	parser.add_argument(
		'-ds', '--data-size', dest='data_size', type=int, default=50000, 
		help='Number of images to load from the dataset.'
	)


	args = parser.parse_args()

	return args

if __name__ == '__main__':
	args = preprocess()

	if args.seed != -1:
		torch.manual_seed(seed)
		np.random.seed(seed)

	if args.use_cuda:
		device = 'cuda:0'
		torch.cuda.set_device(0)
	else:
		device = 'cpu'

	d = HandwrittenDataset('../nist_19_28/', max_load=args.data_size)

	t_in, t_out, v_in, v_out = d.configure(
		split=args.train_val_split, device=device
	)

	network = Net5(
		28, len(d.class_meta.keys()), args.dropout_p
	).to(device)

	print(network)

	trainer = Controller(
		(t_in, t_out, v_in, v_out),
		network,
		seed=(None if args.seed == -1 else args.seed),
		optimizer=args.optimizer,
		lr=args.learning_rate,
		momentum=args.momentum,
		l2rs=args.l2_reg_strength,
		sched=args.use_sched,
		max_iter=args.iterations
	)

	print("Summary:")
	print("\tTraining Iterations: %d"%args.iterations)
	print("\tSplit:               %f"%args.train_val_split)
	print("\tDevice:              %s"%device)
	#print("\tNetwork Parameters:  %d"%network.get_n_params())
	print("\tDataset Size:        %d"%(t_in.shape[0] + v_in.shape[0]))
	print("\tFeature Size:        %d"%t_in.shape[1])
	print("\tClasses:             %d"%len(d.class_meta.keys()))
	print("\tInputs:              %d"%t_in.shape[0])
	print("\tInput Data Size:     %d"%(t_in.shape[0] * t_in.shape[1]))
	print("\tOptimizer:           %s"%args.optimizer)
	print("\tLearning Rate:       %f"%args.learning_rate)
	print("\tL2 Regularization:   %f"%args.l2_reg_strength)
	print("\tScheduler:           %s"%('on' if args.use_sched else 'off'))
	print("\tDropout Probability: %f"%args.dropout_p)

	train_acc, val_acc, cel, timing = trainer.train()

	print("Average Feed Forward on %d images was %fs"%(
		t_in.shape[0], np.array(timing).mean() / 1e9
	))

	fig, (ax1, ax2) = plt.subplots(2, 1)

	celp,   = ax1.plot(range(len(cel)), cel)
	trainp, = ax2.plot(range(len(train_acc)), train_acc)
	valp,   = ax2.plot(range(len(val_acc)), val_acc)

	ax2.legend([trainp, valp], ['Training Accuracy', 'Validation Accuracy'])

	ax1.set_title("Cross Entropy Loss")
	ax2.set_title("Classification Accuracy")

	plt.tight_layout()
	plt.show()