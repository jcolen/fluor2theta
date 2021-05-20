import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from base_predictor import BasePredictor

class fcn_resnet50(BasePredictor):
	def __init__(self, **kwargs):
		super(fcn_resnet50, self).__init__()
		self.model = torchvision.models.segmentation.fcn_resnet50(num_classes=3, pretrained=False)
		self.name = 'fcn_resnet50'
		self.loss = torch.nn.CrossEntropyLoss()

	def forward(self, x):
		return self.model(x.repeat(1, 3, 1, 1))['out']
	
	def getxy(self, batch):
		return batch['x'], batch['label']
	
	def show_defects(self, ax, x, y):
		if len(x.shape) == 3:
			x = x[0]
		ax.clear()
		ax.set_xticks([])
		ax.set_yticks([])
		#Labels are 0 -> -1/2, 1 -> 0, 1 -> +1/2
		ax.imshow(x, cmap='Greys')
		rgba = np.zeros(x.shape + (4,))
		rgba[y == 0, :] = [1., 0, 0, 0.8]
		rgba[y == 2, :] = [0, 0.5, 1., 0.8]
		ax.imshow(rgba)

	def predict_plot(self, batch, **kwargs):
		with torch.no_grad():
			x, y0 = self.getxy(batch)
			y = torch.argmax(self(x), dim=1).cpu().numpy()
			x = x.cpu().numpy()
			y0 = y0.cpu().numpy()

		nplots = min(y0.shape[0], 4)
		fig, ax = plt.subplots(nplots, 2)
		ax[0, 0].set_title('Target')
		ax[0, 1].set_title('ML')
		for i in range(nplots):
			self.show_defects(ax[i, 0], x[i], y0[i])
			self.show_defects(ax[i, 1], x[i], y[i])
		plt.tight_layout()
		return fig

class fcn_resnet101(fcn_resnet50):
	def __init__(self, **kwargs):
		BasePredictor.__init__(self)
		self.name = 'fcn_resnet101'
		self.model = torchvision.models.segmentation.fcn_resnet101(num_classes=3, pretrained=False)
		self.loss = torch.nn.CrossEntropyLoss()
