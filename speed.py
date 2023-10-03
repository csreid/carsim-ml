import click
from tqdm import tqdm
from torch.nn import Sequential
import torch
from models.speed_model import SpeedModel
from datasets.speed_dataset import SDDataset
from trainers.speed_estimate import train

@click.command()
@click.option('--epochs', default=1, help='Number of epochs', type=int)
@click.option('--batch-size', default=1, help='Size of batch', type=int)
@click.option('--path', default=None, help='Path to training data directory', required=True)
@click.option('--gpu/--no-gpu', default=True, help='Train on the GPU?')
@click.option('--output-file', default=None, help='Path to save the output weights')
@click.option('--resume-from', default=None, help='Saved weights from which to resume training')
@click.option('--pretrained/--no-pretrained', default=True, help='Use pretrained resnet weights for input?')
@click.option('--img-features', default=64, help='Number of features in the image latent space', type=int)
def speed_estimator(epochs, batch_size, path, img_features, gpu, output_file, pretrained, resume_from):
	model = SpeedModel(
		img_features,
		pretrained,
		dev='cpu' if not gpu else 'cuda:0'
	)

	data = SDDataset(path, context_len=20)

	model = train(
		model,
		data,
		epochs=epochs,
		batch_size=batch_size,
		dev='cuda:0' if gpu else 'cpu'
	)

	if output_file:
		torch.save(model.state_dict(), output_file)

if __name__ == '__main__':
	speed_estimator()
