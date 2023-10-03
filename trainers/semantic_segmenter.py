import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, draw_segmentation_masks
from torch.nn import CrossEntropyLoss, Sequential, functional as F
import matplotlib.pyplot as plt

writer = SummaryWriter()
colors = [
	(255, 0, 0), #    Roads        =    1u,
	(0, 0, 255), #    Sidewalks    =    2u,
	(0, 255, 0), #    Buildings    =    3u,
	(127,255,212), #    Walls        =    4u,
	(255, 255, 255), #    Fences       =    5u,
	(255, 255, 255), #    Poles        =    6u,
	(255, 0, 127), #    TrafficLight =    7u,
	(255, 0, 127), #    TrafficSigns =    8u,
	(255,255,255), #    Vegetation   =    9u,
	(255,255,255), #    Terrain      =   10u,
	(0,0,128), #    Sky          =   11u,
	(255,255,255), #    Pedestrians  =   12u,
	(255,255,255), #    Rider        =   13u,
	(255,127,0), #    Car          =   14u,
	(255,127,0), #    Truck        =   15u,
	(255,127,0), #    Bus          =   16u,
	(255,127,0), #    Train        =   17u,
	(255,127,0), #    Motorcycle   =   18u,
	(255,127,0), #    Bicycle      =   19u,
	(255,255,255), #    Static       =   20u,
	(255,255,255), #    Dynamic      =   21u,
	(255,255,255), #    Other        =   22u,
	(255,255,255), #    Water        =   23u,
	(127,127,255), #    RoadLines    =   24u,
	(255,255,255), #    Ground       =   25u,
	(255,255,255), #    Bridge       =   26u,
	(255,255,255), #    RailTrack    =   27u,
	(255,255,255), #    GuardRail    =   28u,
	(255,255,255), #    Any          =  0xFF
]

def epoch_log(model, dataset, epoch):
	loader = DataLoader(dataset, batch_size=1000, shuffle=True)
	X, _ = next(iter(loader))
	X = X.to('cuda:0')
	emb = model[0](X)

	writer.add_embedding(emb, label_img=X, tag='Embeddings', global_step=epoch)

def log(model, loss, dataset, ctr):
	loader = DataLoader(dataset, batch_size=4, shuffle=True)
	X, true_seg = next(iter(loader))
	X = X.to('cuda:0')

	Y_pred = torch.argmax(model(X), dim=1).to('cuda:0')

	all_imgs = []
	truths = []
	for idx, (img, ground) in enumerate(zip(X, true_seg)):
		masks = []
		for i in range(28):
			class_mask = (Y_pred[idx] == i)

			masks.append(class_mask)

		mask = torch.stack(masks).bool()
		img = (img * 255).byte()

		masked_Y_pred = draw_segmentation_masks(img.to('cpu'), mask.to('cpu'), alpha=0.5, colors=colors)
		all_imgs.append(masked_Y_pred)

	pred_imgs = torch.stack(all_imgs)
	logged_img = make_grid(all_imgs, nrow=2)

	writer.add_scalar('loss/segmentation', loss, ctr)
	writer.add_image('Segmented image', logged_img, ctr)

def train(model, dataset, epochs, batch_size, dev='cpu', log=log):
	loader = DataLoader(dataset, batch_size, shuffle=True)
	model = model.to(dev)
	opt = Adam(model.parameters())
	loss_fn = CrossEntropyLoss()
	ctr = 0

	total_prog = tqdm(range(epochs), position=0, total=epochs)
	for epoch in total_prog:
		epoch_prog = tqdm(loader, total=int(len(dataset) / batch_size), position=1, leave=False)

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

	return model
