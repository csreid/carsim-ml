import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

import click
from tqdm import tqdm
from torch.nn import Sequential
import torch

@click.command()
@click.option('--epochs', default=1, help='Number of epochs', type=int)
@click.option('--batch-size', default=1, help='Size of batch', type=int)
@click.option('--path', default=None, help='Path to training data directory', required=True)
@click.option('--img-features', default=64, help='Number of features in the image latent space', type=int)
@click.option('--gpu/--no-gpu', default=True, help='Train on the GPU?')
@click.option('--output-file', default=None, help='Path to save the output weights')
@click.option('--pretrained/--no-pretrained', default=False, type=bool, help='Use pretrained resnet weights?')
def autoencoder(epochs, batch_size, path, img_features, gpu, output_file, pretrained):
	from models.vision_input import VisionInput
	from models.vision_reconstructor import VisionReconstructor
	from trainers.reconstructor import train
	from datasets.img_dataset import ImageDataset

	model = Sequential(
		VisionInput(img_features, pretrained=pretrained),
		VisionReconstructor(img_features)
	)

	data = ImageDataset(path)

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
	autoencoder()
