import os
import re
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import dataset.data_processing as dp

import warnings
warnings.filterwarnings("ignore")

def get_folder_info(path):
	fname = os.path.basename(os.path.normpath(path)).lower()

	#Tokenize path around numbers and remove delimiters
	toks = fname.split('_')
	label = {}
	curr, value = None, None
	for tok in toks:
		try:
			value = float(tok)
			label[curr] = value
			continue
		except:
			curr = tok
	if len(label.keys()) == 0:
		return None
	return label

'''
Dataset to hold active nematics information
'''
class NematicsDataset(Dataset):
	'''
	Arguments:
		root_dir(string) - The xy to look for files
		prefix(string) - The file prefix to use
		frames_per_seq(int) - Sequence length to load
		transform(callable, optional) - Optional transform to be applied to a sample
	'''
	def __init__(self, 
				 root_dir, 
				 transform=None,
				 label_info=get_folder_info,
				 force_load=False,
				 validation_split=0.2):
		self.root_dir = root_dir
		self.transform = transform
		self.label_info = label_info

		self.files_index_name = os.path.join(root_dir, 'index_nxny.csv')
		self.build_files_index(force_load)
		self.folders = self.dataframe.folder.unique()
		self.label_names = self.dataframe.columns.to_list()
		self.label_names.remove('folder')
		self.label_names.remove('idx')
		self.label_names.sort()
		self.dataframe.to_csv(self.files_index_name, index=False)
		self.num_folders = len(self.folders)
		print('Found %d sequences in %d folders' % (len(self), self.num_folders))

		split	= int(np.floor(validation_split * len(self)))
		indices = np.arange(len(self))
		np.random.shuffle(indices)
		self.train_indices, self.test_indices = indices[split:], indices[:split]

	def list_file_indices(self, path):
		idxs = None
		fnames = glob.glob(os.path.join(path, 'nx*'))
		inds = [list(map(int, re.findall(r'\d+', os.path.basename(fname))))[-1] for fname in fnames]
		idxs = inds if idxs is None else np.intersect1d(idxs, inds)
		return np.sort(idxs)

	def build_files_index(self, force_load=False):
		if not force_load and os.path.exists(self.files_index_name):
			self.dataframe = pd.read_csv(self.files_index_name)
			return

		#Build an index of the images in different class subfolders
		folders, idxs = [], np.zeros(0, dtype=int)
		labels = {}
		for subdir in os.listdir(self.root_dir):
			dirpath = os.path.join(self.root_dir, subdir)
			if not os.path.isdir(dirpath):
				continue
			
			inds = self.list_file_indices(dirpath)
			if inds is None:
				continue
			
			label = self.label_info(dirpath)
			print('%s: Label = %s' % (os.path.basename(dirpath), str(label)))
			if label is None:
				print('No label found for folder %s' % subdir)
				continue


			nimages = len(inds)
			folders += [subdir] * nimages
			idxs = np.append(idxs, inds)
			for key in label:
				if key in labels:
					labels[key] += [label[key]] * nimages
				else:
					labels[key] = [label[key]] * nimages

		self.dataframe = pd.DataFrame(dict({'folder': folders, 'idx': idxs}, **labels))
	
	def get_loader(self, indices, batch_size, num_workers, pin_memory=True):
		sampler = SubsetRandomSampler(indices)
		loader = torch.utils.data.DataLoader(self, 
			batch_size=batch_size,
			num_workers=num_workers,
			sampler=sampler,
			pin_memory=pin_memory)
		return loader

	def get_image(self, idx):
		subdir = os.path.join(self.root_dir, self.dataframe.folder[idx])
		ind = self.dataframe.idx[idx]
		nx = np.loadtxt(os.path.join(subdir, 'nx%d' % ind))
		ny = np.loadtxt(os.path.join(subdir, 'ny%d' % ind))
		return {'nx': nx, 'ny': ny}
	
	def get_label(self, idx):
		label = np.array([self.dataframe[key][idx] for key in self.label_names])
		return label

	def get_average_labels(self):
		return np.array([np.average(self.dataframe[key]) for key in self.label_names])

	def __len__(self):
		return len(self.dataframe)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		sample = self.get_image(idx)
		sample['label'] = self.get_label(idx)

		if self.transform:
			sample = self.transform(sample)

		return sample
