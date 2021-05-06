import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
import numpy as np
import matplotlib.pyplot as plt
from conv_layers import FlatCnnCell, DeCnnCell, CnnCell, DownsampleCell, UpsampleCell
import data_processing as dp

'''
Generic convolutional unet predictor
'''
class UnetPredictor(nn.Module):
	def __init__(self, channels, **kwargs):
		super(UnetPredictor, self).__init__()
		self.channels = channels
		self.name = 'unet_c%s' % (','.join([str(c) for c in channels]))

		self.cells = nn.ModuleList()
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
		return x
	
	def batch_step(self, batch,  criterion, optimizer):
		if self.training:  optimizer.zero_grad()
		x, y0 = self.getxy(batch)
		y = self(x)
		loss = criterion(y0, y)
		if self.training:
			loss.backward()
			optimizer.step()
		return loss.item()

	def get_transform(self, crop_size):
		return transforms.Compose([
			dp.RandomFlip(),
			dp.RandomCrop(crop_size),
			dp.ToTensor()
		])

	def get_criterion(self):
		return F.mse_loss

	def predict_plot(self, batch, cmap='BuPu', savedir='figures'):
		if not os.path.exists(savedir):
			os.mkdir(savedir)

		with torch.no_grad():
			x, y0 = self.getxy(batch)
			y = self(x)
			xval, xvec, y0val, y0vec, yval, yvec = self.convert_xy_visual(x, y0, y)
			nplots = min(y0.shape[0], 4)
			fig, ax = plt.subplots(nplots, 4)
			for i in range(nplots):
				for j in range(4):
					ax[i, j].clear()
					ax[i, j].set_xticks([])
					ax[i, j].set_yticks([])

				vmin, vmax = np.min(y0val[i]), np.max(y0val[i])
				skip = 5
				xidx = (i, slice(None, None, skip), slice(None, None, skip), 0)
				yidx = (i, slice(None, None, skip), slice(None, None, skip), 1)
				y, x = np.mgrid[:xval.shape[1]:skip, :xval.shape[2]:skip]
				ax[i, 0].imshow(xval[i], cmap='viridis')
				ax[i, 0].quiver(x, y, xvec[xidx], xvec[yidx])


				ax[i, 1].imshow(y0val[i], cmap=cmap, vmin=vmin, vmax=vmax)
				ax[i, 1].quiver(x, y, y0vec[xidx], y0vec[yidx])

				ax[i, 2].imshow(yval[i], cmap=cmap, vmin=vmin, vmax=vmax)
				ax[i, 2].quiver(x, y, yvec[xidx], yvec[yidx])

				ax[i, 3].imshow(np.linalg.norm(yvec[i] - y0vec[i], axis=-1), cmap='bwr')
			ax[0, 0].set_title('Input')
			ax[0, 1].set_title('Target')
			ax[0, 2].set_title('Predicted')
			ax[0, 3].set_title('Diff')
			
			plt.tight_layout()
			plt.savefig(os.path.join(savedir, '%s.png' % self.name),
									dpi=200)
			fig.canvas.draw()
			fig.clf()
			plt.close(fig)

class VelocityToMyosinUnet(UnetPredictor):
	def __init__(self, channels, **kwargs):
		super(VelocityToMyosinUnet, self).__init__(channels, **kwargs)
		self.name = 'vel2my_' + self.name

		self.read_in = FlatCnnCell(2, channels[0])
		self.read_out = FlatCnnCell(channels[0], 4)

	def forward(self, x):
		x = self.read_in(x)
		x = super(VelocityToMyosinUnet, self).forward(x)
		x = self.read_out(x)
		return x
	
	def getxy(self, batch):
		return batch['velocity'], batch['myosin']
	
	def convert_xy_visual(self, x, y0, y):
		#Y is the myosin tensor, plot the positive eigenvalue
		y0 = y0.cpu().numpy()
		y0 = y0.transpose(0, 2, 3, 1)
		y0 = y0.reshape(y0.shape[:-1] + (2, 2,))
		y0val, y0vec = np.linalg.eigh(y0)
		y0val = y0val[..., -1]
		y0vec = y0vec[..., -1] * y0val[..., None]
		
		y = y.cpu().numpy()
		y = y.transpose(0, 2, 3, 1)
		y = y.reshape(y.shape[:-1] + (2, 2,))
		yval, yvec = np.linalg.eigh(y)
		yval = yval[..., -1]
		yvec = yvec[..., -1] * yval[..., None]

		#X is the velocity vector, magnitude is its norm
		x = x.cpu().numpy()
		xvec = x.transpose(0, 2, 3, 1)
		xval = np.linalg.norm(xvec, axis=-1)
		
		return xval, xvec, y0val, y0vec, yval, yvec

class MyosinToVelocityUnet(UnetPredictor):
	def __init__(self, channels, **kwargs):
		super(MyosinToVelocityUnet, self).__init__(channels, **kwargs)
		self.name = 'my2vel_' + self.name

		self.read_in = FlatCnnCell(4, channels[0])
		self.read_out = FlatCnnCell(channels[0], 2)

	def forward(self, x):
		x = self.read_in(x)
		x = super(MyosinToVelocityUnet, self).forward(x)
		x = self.read_out(x)
		return x
	
	def getxy(self, batch):
		return batch['myosin'], batch['velocity']
	
	def convert_xy_visual(self, x, y0, y):
		#X is the myosin tensor, plot the positive eigenvalue
		x = x.cpu().numpy()
		x = x.transpose(0, 2, 3, 1)
		x = x.reshape(x.shape[:-1] + (2, 2,))
		xval, xvec = np.linalg.eigh(x)
		xval = xval[..., -1]
		xvec = xvec[..., -1] * xval[..., None]

		#Y is the velocity vector, magnitude is its norm
		y0 = y0.cpu().numpy()
		y0vec = y0.transpose(0, 2, 3, 1)
		y0val = np.linalg.norm(y0vec, axis=-1)

		y = y.cpu().numpy()
		yvec = y.transpose(0, 2, 3, 1)
		yval = np.linalg.norm(yvec, axis=-1)

		return xval, xvec, y0val, y0vec, yval, yvec
