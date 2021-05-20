import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import os
import math
import matplotlib.pyplot as plt
from conv_layers import CnnCell, DeCnnCell, FlatCnnCell, UpsampleCell, DownsampleCell
from base_predictor import BasePredictor
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

class EncoderDecoderPredictor(BasePredictor):
	def __init__(self,
				 channels,
				 mode='bilinear',
				 sample='upsample',
				 **kwargs):
		
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
