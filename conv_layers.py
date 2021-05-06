import torch 
import torch.nn as nn
import torch.nn.functional as F

class CnnCell(nn.Module):
	def __init__(self, in_channel, out_channel, dropout=0.1):
		super(CnnCell, self).__init__()
		
		self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
		self.bn   = nn.BatchNorm2d(out_channel)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = F.tanh(x)
		x = self.dropout(x)
		return x

class DeCnnCell(nn.Module):
	def __init__(self, in_channel, out_channel, dropout=0.1):
		super(DeCnnCell, self).__init__()
		
		self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
		self.bn		= nn.BatchNorm2d(out_channel)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.deconv(x)
		x = self.bn(x)
		x = F.tanh(x)
		x = self.dropout(x)
		return x

'''
Size-preserving CNN cell
'''
class FlatCnnCell(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(FlatCnnCell, self).__init__()
		
		self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
		self.bn   = nn.BatchNorm2d(out_channel)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = F.tanh(x)
		return x

'''
Interpolating/pooling size-changing cells
'''
class UpsampleCell(nn.Module):
	def __init__(self, in_channel, out_channel, method='bilinear', dropout=0.1):
		super(UpsampleCell, self).__init__()

		self.upsample = nn.Upsample(scale_factor=2, mode=method)
		self.conv = FlatCnnCell(in_channel, out_channel)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.conv(x)
		x = self.upsample(x)
		x = self.dropout(x)
		return x

class DownsampleCell(nn.Module):
	def __init__(self, in_channel, out_channel, dropout=0.1):
		super(DownsampleCell, self).__init__()

		self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
		self.conv = FlatCnnCell(in_channel, out_channel)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.conv(x)
		x = self.downsample(x)
		x = self.dropout(x)
		return x
