import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import os
import math
import matplotlib.pyplot as plt
from conv_layers import CnnCell, DeCnnCell, FlatCnnCell, UpsampleCell, DownsampleCell
import data_processing as dp

class UpsampleEncoder(nn.Module):
	def __init__(self, channels, mode='bilinear'):
		super(UpsampleEncoder, self).__init__()
		self.cells = nn.ModuleList()
		self.cells.append(FlatCnnCell(1, channels[0]))
		for i in range(len(channels)-1):
			if channels[i] == channels[i+1]:
				self.cells.append(FlatCnnCell(channels[i], channels[i+1]))
			elif mode == 'conv':
				self.cells.append(DeCnnCell(channels[i], channels[i+1]))
			else:
				self.cells.append(UpsampleCell(channels[i], channels[i+1], method=mode))

	def forward(self, x):
		for i in range(len(self.cells)):
			x = self.cells[i](x)
		return x

class DownsampleDecoder(nn.Module):
	def __init__(self, channels, mode='bilinear'):
		super(DownsampleDecoder, self).__init__()
		self.cells = nn.ModuleList()
		for i in range(1, len(channels)):
			if channels[-i] == channels[-(i+1)]:
				self.cells.append(FlatCnnCell(channels[-i], channels[-(i+1)]))
			elif mode == 'conv':
				self.cells.append(CnnCell(channels[-i], channels[-(i+1)],
					dropout=0.))
			else:
				self.cells.append(DownsampleCell(channels[-i], channels[-(i+1)],
					dropout=0.))

	def forward(self, x):
		for i in range(len(self.cells)):
			x = self.cells[i](x)
		return x

class DownsampleEncoder(nn.Module):
	def __init__(self, channels, mode='linear'):
		super(DownsampleEncoder, self).__init__()
		self.cells = nn.ModuleList()
		self.cells.append(FlatCnnCell(1, channels[0]))
		for i in range(len(channels)-1):
			if channels[i] == channels[i+1]:
				self.cells.append(FlatCnnCell(channels[i], channels[i+1]))
			elif mode == 'conv':
				self.cells.append(CnnCell(channels[i], channels[i+1]))
			else:
				self.cells.append(DownsampleCell(channels[i], channels[i+1]))

	def forward(self, x):
		for i in range(len(self.cells)):
			x = self.cells[i](x)
		return x

class UpsampleDecoder(nn.Module):
	def __init__(self, channels, mode='linear'):
		super(UpsampleDecoder, self).__init__()
		self.cells = nn.ModuleList()
		for i in range(1, len(channels)):
			if channels[-i] == channels[-(i+1)]:
				self.cells.append(FlatCnnCell(channels[-i], channels[-(i+1)]))
			elif mode == 'conv':
				self.cells.append(DeCnnCell(channels[-i], channels[-(i+1)]))
			else:
				self.cells.append(UpsampleCell(channels[-i], channels[-(i+1)], mode))

	def forward(self, x):
		for i in range(len(self.cells)):
			x = self.cells[i](x)
		return x

'''
Generic encoder/decoder based frame predictor
'''

class EncoderDecoderPredictor(nn.Module):
	def __init__(self,
				 channels,
				 recurrent='lstm',
				 mode='bilinear',
				 residual=True,
				 num_recurrents=2,
				 layers_per_recurrent=1,
				 sample='upsample'):
		
		super(EncoderDecoderPredictor, self).__init__()

		self.channels = channels
		self.name = 'predictor_c%s_%s' % (','.join([str(c) for c in channels]), mode)

		if sample == 'upsample':
			self.encoder = UpsampleEncoder(channels, mode)
			self.decoder = DownsampleDecoder(channels, mode)
		elif sample == 'downsample':
			self.encoder = DownsampleEncoder(channels, mode)
			self.decoder = UpsampleDecoder(channels, mode)
		self.name = sample + self.name

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)

		norm = x.norm(p=2, dim=-3, keepdim=True)
		norm = norm + 1. * (norm == 0)
		x = x.div(norm.expand_as(x))

		return x

	def batch_step(self, x, y, criterion, optimizer):
		if self.training:	optimizer.zero_grad()
		preds = self(x)
		loss = criterion(preds, y)
		if self.training:
			loss.backward()
			optimizer.step()
		return loss.item()
	
	def get_criterion(self):
		return F.l1_loss
	
	def get_transform(self, flip=0.5, transpose=0.5, crop_size=48):
		return transforms.Compose([
			dp.ProcessInputs(),
			dp.RandomFlip(flip),
			dp.RandomTranspose(transpose),
			dp.RandomCrop(crop_size),
			dp.ToTensor()])
	
	def plot_angle(self, ax, x, cmap='BuPu', vmin=-1, vmax=1):
		ax.clear()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.imshow(x[0].cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)

	def predict_plot(self, x, y, savedir='figures', cmap='BuPu'):
		with torch.no_grad():
			preds = self(x)
			nplots = min(x.shape[0], 2)
			fig, ax = plt.subplots(nplots, 4, 
								   figsize=((12, 4*nplots)))
			for i in range(nplots):
				ax[i, 0].imshow(x[i, 0].cpu().numpy(), vmin=0, vmax=1)
				self.plot_angle(ax[i, 1], y[i])
				self.plot_angle(ax[i, 2], preds[i])
				self.plot_angle(ax[i, 3], y[i] - preds[i], cmap='bwr', vmin=-2, vmax=2)
			
			ax[0, 0].set_title('Fluorescence')
			ax[0, 1].set_title('Target')
			ax[0, 2].set_title('Prediction')
			ax[0, 3].set_title('Difference')

			plt.tight_layout()
			plt.savefig(os.path.join(savedir, '%s.png' % self.name),
						dpi=200)
			fig.canvas.draw()
			fig.clf()
