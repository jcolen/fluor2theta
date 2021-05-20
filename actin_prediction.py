import numpy as np
import torch
import os
import argparse

from defects import defect_coordinates
from winding import winding
import unet_predictor as up
import matplotlib.pyplot as plt

from skimage import io
from scipy.ndimage import median_filter, gaussian_filter
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean

def load_tif(filename, outlier=3, radius=7, scale=6):
	tif = io.imread(filename)
	if args.crop[0] > 0 or args.crop[1] < 0:
		tif = tif[..., args.crop[0]:args.crop[1], args.crop[0]:args.crop[1]]
	if tif.shape[-1] == 3:
		tif = rgb2gray(tif)

	tif[tif < args.threshold[0]] = args.threshold[0]
	tif[tif > args.threshold[1]] = args.threshold[1]

	y, x = tif.shape[-2:]
	y = y // scale * scale
	x = x // scale * scale
	tif = tif[..., :y, :x]

	output = []
	for i in range(tif.shape[0]):
		d = tif[i]
		d = downscale_local_mean(d, (scale, scale))
		if args.median:
			mean, std = np.mean(d[:]), np.std(d[:])
			outliers = d > (mean + outlier * std)
			d[outliers] = median_filter(d, size=radius)[outliers]
		if args.gaussian:
			#d = gaussian_filter(d, sigma=1.5)
			d = gaussian_filter(d, sigma=1.0)
		d = (d - np.min(d)) / (np.max(d) - np.min(d))
		output.append(d)
	tif = np.array(output)
	tif = (tif - np.min(tif)) / (np.max(tif) - np.min(tif))

	#crop to 128x128
	tif = tif[:, :128, :128]
	
	#cos^2 \theta
	return tif

def plot_angle(ax, theta, cmap='BuPu', quiver=True, wind=False, title=None):
	ax.clear()
	ax.set_xticks([])
	ax.set_yticks([])
	if title:	ax.set_title(title)
	ax.imshow(theta, cmap=cmap)
	if quiver:
		skip = 6
		y, x = np.mgrid[:theta.shape[0]:skip, :theta.shape[1]:skip]
		ax.quiver(x, y, np.cos(theta)[::skip, ::skip], np.sin(theta)[::skip, ::skip])
	if wind:
		radius=2
		coords = defect_coordinates(theta, radius=radius)
		charge = winding(theta, radius=radius)
		charge = charge[coords[:, 1], coords[:, 0]]
		colors = [[1., 0, 0] if abs(c + 0.5) < 0.1 else [0, 0.5, 1.] for c in charge]
		ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=50)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('tif', type=str)
	parser.add_argument('-m', '--model', type=str,
		default='models/unet_c4,4,4,8,8,8,16,16,16,32,32,32,64,64,64,128,128,128')
	parser.add_argument('--crop', type=int, nargs='+', default=[0, 0])
	parser.add_argument('--median', action='store_true')
	parser.add_argument('--gaussian', action='store_false')
	parser.add_argument('--scale', type=int, default=4)
	parser.add_argument('--threshold', type=int, nargs='+', default=[0, 255])
	args = parser.parse_args()
	
	#GPU or CPU
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(device)

	model = up.UnetPredictor(
		channels=[int(c) for c in os.path.basename(args.model).split('_')[-1][1:].split(',')])
	model.load_state_dict(torch.load(args.model)['state_dict'])
	model.to(device)
	model.eval()

	tif = load_tif(args.tif, scale=args.scale)
	print(tif.shape)
	
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 2, figsize=(8, 5))
	fig.canvas.manager.window.wm_geometry('+0+0')
	plt.ion()
	plt.show()

	with torch.no_grad():	
		y = model(torch.tensor(tif[:, None].copy(), dtype=torch.float, device=device))[:, 0].cpu().numpy()
	for i in range(tif.shape[0]):
		plot_angle(ax[0], tif[i], title=r'$\cos^2\theta$', cmap='Greys', quiver=False)
		plot_angle(ax[1], y[i], title=r'$\theta_{ML}$', wind=True)
		if input() == 'q':
			from sys import exit
			exit(0)
