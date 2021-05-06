import torch
import torch.nn as nn
from conv_layers import FlatCnnCell, DownsampleCell, UpsampleCell
from base_predictor import BasePredictor

'''
Generic convolutional unet predictor
'''
class UnetPredictor(BasePredictor):
	def __init__(self, channels, **kwargs):
		super(UnetPredictor, self).__init__()
		self.channels = channels
		self.name = 'unet_c%s' % (','.join([str(c) for c in channels]))

		self.cells = nn.ModuleList()
		self.cells.append(FlatCnnCell(1, channels[0]))
		for i in range(len(channels)-1):
			if channels[i+1] == channels[i]:
				self.cells.append(FlatCnnCell(channels[i], channels[i+1]))
			else:
				self.cells.append(DownsampleCell(channels[i], channels[i+1]))

		for i in range(1, len(channels)):
			if channels[-(i+1)] == channels[-i]:
				self.cells.append(FlatCnnCell(channels[-i], channels[-(i+1)]))
			else:
				self.cells.append(UpsampleCell(channels[-i], channels[-(i+1)]))
				self.cells.append(FlatCnnCell(2*channels[-(i+1)], channels[-(i+1)]))
			

	def forward(self, x):
		encoder_outputs = []
		decoder_idx = -1
		for i, cell in enumerate(self.cells):
			if isinstance(cell, DownsampleCell):
				encoder_outputs.append(x)
			x = cell(x)
			if isinstance(cell, UpsampleCell):
				x = torch.cat([x, encoder_outputs[decoder_idx]], dim=-3)
				decoder_idx -= 1
		
		norm = x.norm(p=2, dim=-3, keepdim=True)
		norm = norm + 1. * (norm == 0)
		x = x.div(norm.expand_as(x))

		return x
