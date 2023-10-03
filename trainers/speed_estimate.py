import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import MSELoss, HuberLoss
import matplotlib.pyplot as plt

writer = SummaryWriter()
def epoch_log(model, dataset, epoch):
	loader = DataLoader(dataset, batch_size=1000, shuffle=True)
	X, _ = next(iter(loader))
	X = X.to('cuda:0')
	emb = model.vision(X)

	writer.add_embedding(emb, label_img=X, tag='Embeddings', global_step=epoch)

def log(model, loss, dataset, step):
	loader = DataLoader(dataset, batch_size=4, shuffle=True)
	sample_l, sample_r, acc, sample_speeds = next(iter(loader))
	speed_est = model(sample_l.to('cuda:0'), sample_r.to('cuda:0'), acc.to('cuda:0')).squeeze()
	sample_speeds = sample_speeds.squeeze()

	fig = plt.figure()

	styles = [
		('r:', 'r'),
		('b:', 'b'),
		('g:', 'g'),
		('m:', 'm')
	]

	fig, ((a1, a2), (a3, a4)) = plt.subplots(ncols=2, nrows=2)
	axes = [a1, a2, a3, a4]
	for idx, (l, r, speeds, est) in enumerate(zip(sample_l, sample_r, sample_speeds, speed_est)):
		axes[idx].plot(
			est.detach().cpu().numpy(),
			styles[idx][0],
			label=f'Estimated Speed',
			alpha=0.6
		)

		axes[idx].plot(
			speeds.detach().cpu().numpy(),
			styles[idx][1],
			label=f'True Speed',
			alpha=0.6
		)
		axes[idx].set_ylim(-1, 1)
		axes[idx].legend()

	writer.add_figure('True Speed vs Estimated Speed', fig, step)
	writer.add_video('Sample Sequence', torch.cat((sample_r, sample_l), dim=4), step)
	writer.add_scalar('Loss/speed', loss, step)

	writer.add_scalar(
		'Difference from mean',
		torch.mean(torch.abs(speed_est - torch.mean(speed_est))),
		step
	)

def train(model, dataset, epochs, batch_size, dev='cpu', checkpoint_on='end'):
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	model = model.to(dev)

	opt = Adam(model.parameters())
	loss_fn = MSELoss()
	ctr = 0
	total_prog = tqdm(range(epochs), position=0, total=epochs)
	for epoch in total_prog:
		epoch_prog = tqdm(loader, total=int(len(dataset) / batch_size), position=1, leave=False)
		for (imgs_l, imgs_r, acc, Y) in epoch_prog:
			imgs_l = imgs_l.to(dev)
			imgs_r = imgs_r.to(dev)
			acc = acc.to(dev)
			Y = Y.to(dev)

			Y_pred = model(imgs_l, imgs_r, acc)

			loss = loss_fn(Y_pred, Y)

			opt.zero_grad()
			loss.backward()
			opt.step()

			log(model, loss, dataset, ctr)
			writer.add_scalar('Loss/speed', loss, ctr)

			ctr += 1
		torch.save(model.state_dict(), 'vision_speed_est.pt')
