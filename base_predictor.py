import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import pytorch_lightning as pl

import os
import numpy as np
import data_processing as dp
import matplotlib.pyplot as plt

'''
Generic predictor abstract class
'''

class BasePredictor(pl.LightningModule):
	def __init__(self, **kwargs):
		super(BasePredictor, self).__init__()
		self.loss = F.mse_loss

	def getxy(self, batch):
		return batch['x'], batch['y']
	
	def training_step(self, batch,  batch_idx):
		x, y0 = self.getxy(batch)
		y = self(x)
		loss = self.loss(y, y0)
		self.log('train_loss', loss)
		return loss
	
	def validation_step(self, batch, batch_idx):
		x, y0 = self.getxy(batch)
		y = self(x)
		loss = self.loss(y, y0)
		self.log('val_loss', loss)
		return {'loss': loss, 'batch': batch}
	
	def validation_epoch_end(self, outs):
		fig = self.predict_plot(outs[0]['batch'])	
		self.logger.experiment.add_figure('example', fig, close=True, global_step=self.trainer.global_step)
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.92)
		return {'optimizer': optimizer,
				'lr_scheduler': scheduler}
	
	def get_transform(self, crop_size, flip=0.5, transpose=0.5):
		return transforms.Compose([
			dp.ProcessInputs(),
			dp.RandomFlip(flip),
			dp.RandomTranspose(transpose),
			dp.RandomCrop(crop_size),
			dp.ToTensor()])
	
	#Visualize model predictions
	
	def convert_xy_visual(self, x, y0, y):
		y0 = y0.cpu().numpy()
		y = y.cpu().numpy()
		x = x.cpu().numpy()

		theta0 = np.arctan2(y0[:, 0], y0[:, 1]) / 2
		theta0[theta0 < 0] += np.pi
		theta0[theta0 > np.pi] -= np.pi

		theta = np.arctan2(y[:, 0], y[:, 1]) / 2
		theta[theta < 0] += np.pi
		theta[theta > np.pi] -= np.pi

		n0 = np.array([np.cos(theta0), np.sin(theta0)])
		n = np.array([np.cos(theta), np.sin(theta)])

		return x[:, 0], theta0, n0, theta, n
	
	def predict_plot(self, batch, cmap='BuPu'):
		with torch.no_grad():
			x, y0 = self.getxy(batch)
			y = self(x)
			xval, y0val, y0vec, yval, yvec = self.convert_xy_visual(x, y0, y)
		nplots = min(y0.shape[0], 2)
		fig, ax = plt.subplots(nplots, 4)
		for i in range(nplots):
			for j in range(4):
				ax[i, j].clear()
				ax[i, j].set_xticks([])
				ax[i, j].set_yticks([])

			vmin, vmax = np.min(y0val[i]), np.max(y0val[i])
			skip = 5
			xidx = (0, i, slice(None, None, skip), slice(None, None, skip))
			yidx = (1, i, slice(None, None, skip), slice(None, None, skip))
			y, x = np.mgrid[:xval.shape[1]:skip, :xval.shape[2]:skip]
			ax[i, 0].imshow(xval[i], cmap='Greys')

			ax[i, 1].imshow(y0val[i], cmap=cmap, vmin=vmin, vmax=vmax)
			ax[i, 1].quiver(x, y, y0vec[xidx], y0vec[yidx])

			ax[i, 2].imshow(yval[i], cmap=cmap, vmin=vmin, vmax=vmax)
			ax[i, 2].quiver(x, y, yvec[xidx], yvec[yidx])

			ax[i, 3].imshow(np.linalg.norm(yvec[:, i] - y0vec[:, i], axis=0), cmap='Reds', vmin=0, vmax=1)
		ax[0, 0].set_title('Input')
		ax[0, 1].set_title('Target')
		ax[0, 2].set_title('Predicted')
		ax[0, 3].set_title('Diff')
		
		plt.tight_layout()
		return fig
