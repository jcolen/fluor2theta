import matplotlib 
matplotlib.use('Agg')
import argparse
import os
import numpy as np
import pandas as pd
from time import time

import torch
from torchvision import transforms

from datasets import NematicsDataset
import data_processing as dp
import encoder_decoder_predictor as edp
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.bottom'] = False
matplotlib.rcParams['xtick.labelsize'] = 0
matplotlib.rcParams['ytick.left'] = False
matplotlib.rcParams['ytick.labelsize'] = 0
matplotlib.rcParams['axes.labelsize'] = 5
matplotlib.rcParams['axes.titlesize'] = 5

def get_model(args):
	kwargs = {
		'channels': args.channels,
		'mode': args.mode,
		'sample': args.sample,
	}
	return edp.EncoderDecoderPredictor(**kwargs)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	
	#Training parameters
	parser.add_argument('-d', '--directory', type=str, default='/home/jcolen/data/short_time_multi_parameter')
	parser.add_argument('--patient', type=int, default=100)
	parser.add_argument('-b', '--batch_size', type=int, default=2)
	parser.add_argument('--crop_size', type=int, default=64)

	#NN parameters
	parser.add_argument('--sample', choices=['upsample', 'downsample'], default='upsample')
	parser.add_argument('--mode', choices=['bilinear', 'conv'], default='bilinear')
	parser.add_argument('-c', '--channels', type=int, nargs='+', default=[2,4,6])

	args = parser.parse_args()

	#GPU or CPU
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	pin_memory = True
	print(device)
	
	# Model
	model = get_model(args)
	criterion = model.get_criterion()

	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.92)
	print(model.name)

	# Dataset
	dataset = NematicsDataset(args.directory,
		transform=model.get_transform(crop_size=args.crop_size))
	loader = dataset.get_loader(dataset.train_indices, args.batch_size, 2, pin_memory)
	batch = next(iter(loader))

	print('Beginning training', flush=True)
	

	#Training
	patient = args.patient
	best_epoch, epoch = 0, -1
	loss_min = np.Inf
	while True:
		epoch += 1
		if epoch - best_epoch >= patient:
			print('early stop at epoch %g' % best_epoch)
			break
		
		t_ini = time()
		loss = model.train().batch_step(batch['x'].to(device),
										batch['y'].to(device),
										criterion, optimizer)
		t_end = time()
		print('Epoch %g: loss: %g\ttime=%g' % \
			(epoch, loss, t_end - t_ini), flush=True)
		scheduler.step(loss)
		if loss < loss_min:
			model.eval().predict_plot(batch['x'].to(device), batch['y'].to(device))
			best_epoch = epoch
			loss_min = loss
