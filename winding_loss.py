import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss

PI = np.pi
PIHALF = PI / 2.
PI2 = np.pi * 2.

'''
Winding loss takes two angle fields and computes the difference in the heatmap of topological charge. This heatmap is computed by computing the local winding number at each point in the grid
'''
class WindingLoss(_Loss):
	def __init__(self, radius=2, method='mse'):
		super(WindingLoss, self).__init__(radius, method)
		if method == 'mse':
			self.method =  F.mse_loss
		elif method == 'l1':
			self.method = F.l1_loss
		self.radius = radius

	def forward(self, input, target):
		winding_input = winding(input, self.radius)
		winding_target = winding(target, self.radius)
		return self.method(winding_input, winding_target, reduction='sum')

def winding(theta, radius=1):
	gy, gx = torch.empty_like(theta), torch.empty_like(theta)
	gy[..., :-1, :] = theta[..., 1:, :] - theta[..., :-1, :]
	gy[..., -1, :] = theta[..., -1, :] - theta[..., -2, :]
	gx[..., :-1] = theta[..., 1:] - theta[..., :-1]
	gx[..., -1] = theta[..., -1] - theta[..., -2]

	gy[gy < -PIHALF] += PI
	gy[gy > PIHALF] -= PI
	gx[gx < -PIHALF] += PI
	gx[gx > PIHALF] -= PI	

	wind = torch.zeros_like(gx)
	r2 = 2 * radius
	for i in range(r2):
		bnd = r2-i
		wind[..., radius:-radius, radius:-radius] -= gx[..., r2:, i:-bnd]
		wind[..., radius:-radius, radius:-radius] += gx[..., :-r2, i:-bnd]
		wind[..., radius:-radius, radius:-radius] += gy[..., i:-bnd, r2:]
		wind[..., radius:-radius, radius:-radius] -= gy[..., i:-bnd, :-r2]
	wind /= PI2
	return wind

if __name__=='__main__':
	charges = np.loadtxt('figures/ch700')
	nx = np.loadtxt('figures/nx700')
	ny = np.loadtxt('figures/ny700')

	theta = np.arctan2(ny, nx)
	theta[theta < 0] += np.pi
	theta[theta > np.pi] -= np.pi

	wind = winding(torch.tensor(theta)).numpy()

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(charges)
	ax[1].imshow(wind)
	plt.show()

