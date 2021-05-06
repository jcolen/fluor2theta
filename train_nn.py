import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
from time import time
from torchvision import transforms

import data_processing as dp
from datasets import NematicsSequenceDataset
from models import TimeEvolver

class WeightedJointLoss(nn.MSELoss):
	def __init__(self, size_average=None, reduction: str='mean', 
				 beta=0.0001) -> None:
		super(WeightedJointLoss, self).__init__(size_average, reduction)
		self.reduction = reduction
		self.beta = beta
	
	def forward(self, ypred, y, x, logvar):
		yloss = F.mse_loss(ypred, y)
		vaeloss = 0.5 * torch.mean(torch.sum(x * x + logvar.exp() - logvar - 1, dim=-1))
		return yloss + self.beta * vaeloss

class ModelContainer(object):
	def __init__(self, device):
		super(ModelContainer, self).__init__()
		self.optimizers = []
		self.schedulers = []
		self.loss_mins = []
		self.losses = []
		self.best_epochs = []
		self.models = []
		self.device = device
	
	def train(self):
		for model in self.models:
			model.train()
		return self

	def eval(self):
		for model in self.models:
			model.eval()
		return self

	def __getitem__(self, idx):
		return self.models[idx]
	
	def __len__(self):
		return len(self.models)

	def add_model(self, model, force_new=False):
		if not force_new and os.path.exists('models/%s' % model.name):
			print('Loading model state dict from file')
			model_info = torch.load('models/%s' % model.name)
			model.load_state_dict(model_info['state_dict'])
			self.loss_mins.append(model_info['loss'])
			self.losses.append(model_info['losses'])
		else:
			self.loss_mins.append(np.Inf)
			self.losses.append([])
		self.models.append(model.to(self.device))
		self.optimizers.append(torch.optim.Adam(model.parameters(), lr=0.001))
		self.schedulers.append(torch.optim.lr_scheduler.ExponentialLR(self.optimizers[-1], 0.92))
		self.best_epochs.append(len(self.losses[-1]))
		print(model.name)

def iterate_loader(models, loader, criterion):
	losses = [0,] * len(models)
	for i, batch in enumerate(loader):
		for j in range(len(models)):
			losses[j] += models[j].batch_step(batch, criterion, models.optimizers[j], 
				device=models.device, depth=args.depth)
		if args.trial and i == 2:	break
	return [loss / i for loss in losses]

def predict(model, device, loader, outfile):
	model.eval()
	with open(outfile, 'w') as fout:
		with torch.no_grad():
			for cnt, batch in enumerate(loader):
				labels, preds = model.batch_predict_encoder(batch, device)
				for i in range(labels.shape[0]):
					for j in range(labels.shape[1]):
						fout.write('%g\t' % labels[i, j])
					for j in range(preds.shape[1]):
						fout.write('%g\t' % preds[i, j])
					fout.write('\n')
				if args.trial and cnt == 2:	break

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, 
		default='/home/jcolen/data/short_time_multi_parameter/') 
	parser.add_argument('-r', '--random_crop', type=int, default=32, 
		help='Crop size of image')
	parser.add_argument('-f', '--num_frames', type=int, default=8, 
		help='Number of frames')
	parser.add_argument('-c', '--encoder_channels', type=str, nargs='+', default=['1,16,16,32,32'])
	parser.add_argument('-k', '--kernel_size', type=int, default=3)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--beta', type=float, default=0.0001)
	parser.add_argument('--depth', type=int, default=2)
	parser.add_argument('--extra_latents', type=int, default=2)
	parser.add_argument('-e', '--epochs', type=int, default=100)
	parser.add_argument('-b', '--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=6)
	parser.add_argument('--validation_split', type=float, default=0.2)
	parser.add_argument('--force_new', action='store_true')
	parser.add_argument('--trial', action='store_true')
	args = parser.parse_args()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	pin_memory = True
	print(device)

	#Dataset
	transform = transforms.Compose([
		dp.Sin2t(),
		dp.RandomTranspose(),
		dp.RandomFlip(),
		dp.RandomCrop(args.random_crop),
		dp.AverageTimeLabel(),
		dp.SwapTimeChannelAxes(),
		dp.ToTensor()])
	dataset = NematicsSequenceDataset(args.dir, args.num_frames, transform=transform, 
		validation_split=args.validation_split)
	train_loader = dataset.get_loader(dataset.train_indices, args.batch_size, args.num_workers, pin_memory)
	test_loader = dataset.get_loader(dataset.test_indices, args.batch_size, args.num_workers, pin_memory)

	models = ModelContainer(device)
	optimizers, schedulers = [], []
	for ch in args.encoder_channels:
		models.add_model(TimeEvolver(
			encoder_channels=[int(c) for c in ch.split(',')],
			extra_latents=args.extra_latents,
			params=dataset.label_names,
			num_frames=args.num_frames,
			pooling=True))
		#models.add_model(TimeEvolver(
		#	encoder_channels=[int(c) for c in ch.split(',')],
		#	params=dataset.label_names,
		#	num_frames=args.num_frames,
		#	pooling=False))
	criterion = WeightedJointLoss(beta=args.beta)

	#Training
	patient = 20
	for epoch in range(args.epochs):
		flag = True
		for best_epoch in models.best_epochs:
			if epoch - best_epoch < patient:
				flag = False
		if flag:
			print('early stop at epoch %g'%best_epoch)
			break
		
		t_ini = time()
		loss_trains = iterate_loader(models.train(), train_loader, criterion)
		loss_tests	= iterate_loader(models.eval(), test_loader, criterion)
		t_end = time()
		
		print('Epoch %g: time=%g' % (epoch, t_end - t_ini))
		for i in range(len(models)):
			models.schedulers[i].step()
			print('\tloss_train=%g, loss_test=%g' % (loss_trains[i], loss_tests[i]), flush=True)
			if loss_tests[i] < models.loss_mins[i]:
				models.loss_mins[i] = loss_tests[i]
				torch.save(
					{'state_dict': models[i].state_dict(),
					 'loss': loss_tests[i],
					 'losses': models.losses[i],
					 'beta': args.beta,
					 'depth': args.depth},
					 'models/%s' % models[i].name)
				predict(models[i], device, test_loader, 'predictions/%s' % models[i].name)
				models.best_epochs[i] = epoch
