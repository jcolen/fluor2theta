import os
import torch
import numpy as np
import random
import scipy 
import skimage.transform

from torchvision import transforms

from winding import winding

import warnings
warnings.filterwarnings("ignore")

class ProcessInputs(object):
	def __call__(self, sample):
		nx, ny = sample['nx'], sample['ny']
		cos_squared = nx * nx
		sin2t = 2 * nx * ny
		cos2t = nx * nx - ny * ny
		theta = np.arctan2(ny, nx)
		theta[theta < 0] += np.pi
		theta[theta > np.pi] -= np.pi
		wind = winding(theta, radius=2)
		label = (2 * wind + 1).astype(int)
		return {'x': cos_squared[None],
				'y': np.stack((sin2t, cos2t)),
				'theta': theta[None],
				'winding': wind[None],
				'label': label}
		
class RandomCrop(object):
	def __init__(self, crop_size, ndims=2):
		self.ndims = ndims
		
		assert isinstance(crop_size, (int, tuple))
		if isinstance(crop_size, int):
			self.crop_size = (crop_size, ) * self.ndims
		else:
			assert len(crop_size) == self.ndims
			self.crop_size = crop_size

	def __call__(self, sample):
		x = sample['x']
		
		dims = x.shape[-self.ndims:]
		corner = [np.random.randint(0, d-nd) for d, nd in zip(dims, self.crop_size)]

		crop_indices = tuple(np.s_[c:c+nd] for c, nd in zip(corner, self.crop_size))
		for key in sample.keys():
			same_indices = tuple(np.s_[0:d] for d in sample[key].shape[:-self.ndims])
			indices = same_indices + crop_indices
			sample[key] = sample[key][indices]
		
		return sample

class RandomTranspose(object):
	def __init__(self, prob=0.5, ndims=2):
		self.prob  = prob
		self.ndims = ndims

	def __call__(self, sample):
		if np.random.random() < self.prob:
			axes = random.sample(range(-self.ndims, 0), 2)
			for key in sample.keys():
				sample[key] = np.swapaxes(sample[key], axes[0], axes[1])
			
		return sample
	
class RandomFlip(object):
	def __init__(self, prob=0.5, ndims=2):
		self.prob  = prob
		self.ndims = ndims

	def __call__(self, sample):
		for dim in range(-self.ndims, 0):
			if np.random.random() < self.prob:
				for key in sample.keys():
					sample[key] = np.flip(sample[key], axis=dim)
				
		return sample

class ToTensor(object):
	def __call__(self, sample):
		for key in sample.keys():
			sample[key] = torch.tensor(sample[key].copy(), 
				dtype=torch.int64 if sample[key].dtype == int else torch.float32)
		return sample
