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
import unet_predictor as up
import fcn_resnet_wrapper as fcn
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.bottom'] = False
matplotlib.rcParams['xtick.labelsize'] = 0
matplotlib.rcParams['ytick.left'] = False
matplotlib.rcParams['ytick.labelsize'] = 0
matplotlib.rcParams['axes.labelsize'] = 5
matplotlib.rcParams['axes.titlesize'] = 5

preds_dict = {
	'basic': edp.EncoderDecoderPredictor,
	'unet': up.UnetPredictor,
	'r50': fcn.fcn_resnet50,
	'r101': fcn.fcn_resnet101,
	'upy': fcn.unet_pytorch
}

def get_model(args):
	kwargs = {
		'channels': args.channels,
		'mode': args.mode,
		'sample': args.sample,
	}
	return preds_dict[args.predictor](**kwargs)

def iterate_loader(model, loader, optimizer, criterion, device):
	loss = 0
	for i, batch in enumerate(loader):
		for key in batch:
			batch[key] = batch[key].to(device)
		loss += model.batch_step(batch, criterion, optimizer)
	return loss / i

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	
	#Training parameters
	parser.add_argument('-d', '--directory', type=str, default='/home/jcolen/data/short_time_multi_parameter')
	parser.add_argument('--patient', type=int, default=100)
	parser.add_argument('-b', '--batch_size', type=int, default=32)
	parser.add_argument('--validation_split', type=float, default=0.2)
	parser.add_argument('--num_workers', type=int, default=3)
	parser.add_argument('--crop_size', type=int, default=64)

	#NN parameters
	parser.add_argument('-p', '--predictor', choices=preds_dict.keys(), default='basic')
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
	loss_min = np.Inf
	losses = []
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.92)
	print(model.name)

	# Dataset
	dataset = NematicsDataset(args.directory,
		validation_split=args.validation_split,
		transform=model.get_transform(crop_size=args.crop_size))
	loader = dataset.get_loader(dataset.train_indices, args.batch_size, 2, pin_memory)
	train_loader = dataset.get_loader(dataset.train_indices, args.batch_size, 
									  args.num_workers, pin_memory)
	test_loader = dataset.get_loader(dataset.test_indices, args.batch_size, 
									 args.num_workers, pin_memory)

	#Training
	patient = args.patient
	best_epoch, epoch = 0, -1
	while True:
		epoch += 1
		if epoch - best_epoch >= patient:
			print('early stop at epoch %g' % best_epoch)
			break
		
		t_ini = time()
		loss_train = iterate_loader(model.train(), train_loader, optimizer, criterion, device)
		with torch.no_grad():
			loss_test = iterate_loader(model.eval(), test_loader, optimizer, criterion, device)
		t_end = time()
		print('Epoch %g: train: %g, test: %g\ttime=%g' % \
			(epoch, loss_train, loss_test, t_end - t_ini), flush=True)
		scheduler.step(loss_test)
		losses.append(loss_test)
		if loss_test < loss_min:
			batch = next(iter(test_loader))
			for key in batch:
				batch[key] = batch[key].to(device)
			model.eval().predict_plot(batch)
			torch.save(
				{'state_dict': model.state_dict(),
				 'loss': loss_test,
				 'losses': losses},
				 'models/%s' % (model.name))
			best_epoch = epoch
			loss_min = loss_test
