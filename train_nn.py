import matplotlib 
matplotlib.use('Agg')
import argparse
import pytorch_lightning as pl

from datasets import NematicsDataset
import data_processing as dp
import encoder_decoder_predictor as edp
import unet_predictor as up
import fcn_resnet_wrapper as fcn

preds_dict = {
	'basic': edp.EncoderDecoderPredictor,
	'unet': up.UnetPredictor,
	'r50': fcn.fcn_resnet50,
	'r101': fcn.fcn_resnet101,
}

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	#Training parameters
	parser.add_argument('-d', '--directory', type=str, default='/home/jcolen/data/short_time_multi_parameter')
	parser.add_argument('-b', '--batch_size', type=int, default=32)
	parser.add_argument('--validation_split', type=float, default=0.2)
	parser.add_argument('--num_workers', type=int, default=3)
	parser.add_argument('--crop_size', type=int, default=64)

	#NN parameters
	parser.add_argument('-p', '--predictor', choices=preds_dict.keys(), default='basic')
	parser.add_argument('--sample', choices=['upsample', 'downsample'], default='upsample')
	parser.add_argument('--mode', choices=['bilinear', 'conv'], default='bilinear')
	parser.add_argument('-c', '--channels', type=int, nargs='+', default=[2,4,6])
	parser = pl.Trainer.add_argparse_args(parser)
	args = parser.parse_args()

	# Model
	model = preds_dict[args.predictor](**vars(args))
	print(model.name)

	# Dataset
	dataset = NematicsDataset(args.directory,
		validation_split=args.validation_split,
		transform=model.get_transform(crop_size=args.crop_size))
	train_loader = dataset.get_loader(dataset.train_indices, args.batch_size, 
									  args.num_workers)
	test_loader = dataset.get_loader(dataset.test_indices, args.batch_size, 
									 args.num_workers)

	logger = pl.loggers.TensorBoardLogger('tb_logs', name=model.name)
	trainer = pl.Trainer.from_argparse_args(args)
	trainer.logger = logger
	trainer.log_every_n_steps = min(len(train_loader), 50)
	trainer.fit(model, train_loader, test_loader)
