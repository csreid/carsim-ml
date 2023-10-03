import time
import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import MSELoss, HuberLoss, Sequential
import matplotlib.pyplot as plt

writer = SummaryWriter()

def epoch_log(model, dataset, epoch):
	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	X, _ = next(iter(loader))
	X = X.to('cpu')#.to('cuda:0')
	emb = model.to('cpu')[0](X)

	writer.add_embedding(emb, label_img=X, tag='Embeddings', global_step=epoch)

	model.to('cuda:0')

def log(model, loss, dataset, ctr):
	loader = DataLoader(dataset, batch_size=4, shuffle=True)
	imgs, _ = next(iter(loader))

	imgs = imgs.to('cuda:0')
	recon_imgs = model(imgs.to('cuda:0'))

	imgs = torch.cat((imgs, recon_imgs), dim=2)

	writer.add_images('Reconstructed Images', imgs, ctr)
	writer.add_scalar('loss/reconstruction', loss, ctr)

def train(model, dataset, epochs, batch_size, dev='cpu', log=log):
	loader = DataLoader(dataset, batch_size, shuffle=True)
	model = model.to(dev)
	opt = Adam(model.parameters())
	loss_fn = MSELoss()
	ctr = 0

	total_prog = tqdm(range(epochs), position=0, total=epochs)
	for epoch in total_prog:
		epoch_prog = tqdm(loader, total=int(len(dataset) / batch_size), position=1, leave=False)
		epoch_log(model, dataset, epoch)

		for X, Y in epoch_prog:
			X = X.to(dev)
			Y = Y.to(dev)

			Y_pred = model(X).to(dev)

			loss = loss_fn(Y_pred, Y)

			opt.zero_grad()
			loss.backward()
			opt.step()

			if log:
				log(model, loss, dataset, ctr)

			ctr += 1


		torch.save(model.state_dict(), 'vision_reconstructor.pt')

	return model
