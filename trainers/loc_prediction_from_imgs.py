import time
import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

import torch
from tqdm import tqdm
from loc_est_dataset import LocationDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import MSELoss, HuberLoss, Sequential, Linear
from vision_input import VisionInput
from vision_reconstructor import VisionReconstructor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

writer = SummaryWriter()
dataset = LocationDataset('data/run_2023_09_18_20_49_33/')

def analyze(model, dataset, desc):
	model = model.to('cuda:0')
	loader = DataLoader(dataset, batch_size=512, shuffle=True)
	all_embs = []
	pts = []
	embedding_progress = tqdm(loader)
	embedding_progress.set_description(desc)
	for X, pt in embedding_progress:
		embs = model[0](X.to('cuda:0'))
		all_embs.append(embs)
		pts.append(pt)

	all_embs = torch.cat(all_embs, dim=0).cpu().detach().numpy()
	pts = torch.cat(pts, dim=0).cpu().detach().numpy()

	reg = LinearRegression(fit_intercept=False)
	reg.fit(all_embs, pts)

	preds = reg.predict(all_embs)
	r2_x = r2_score(preds[:, 0], pts[:, 0])
	r2_y = r2_score(preds[:, 1], pts[:, 1])
	tqdm.write(f'R2 value (X-coordinate): {r2_x}')
	tqdm.write(f'R2 value (Y-coordinate): {r2_y}')

	fig, (ax1, ax2) = plt.subplots(ncols=2)
	ax1.plot(reg.predict(all_embs)[:400, 0], pts[:400, 0], 'ro', alpha=0.2)
	ax1.set_ylabel('True X-value')
	ax1.set_xlabel('X-value predicted from LR')

	ax2.plot(reg.predict(all_embs)[:400, 1], pts[:400, 1], 'ro', alpha=0.2)
	ax2.set_ylabel('True Y-value')
	ax2.set_xlabel('Y-value predicted from LR')
	plt.show()

untrained = Sequential(
	VisionInput(256, network=None),
	VisionReconstructor(256)
)

just_resnet = Sequential(
	VisionInput(256),
	VisionReconstructor(256)
)

trained = Sequential(
	VisionInput(256),
	VisionReconstructor(256)
)
trained.load_state_dict(torch.load('vision_reconstructor.pt'))

for model, desc in zip(
		[untrained, just_resnet, trained],
		['Embedding images with no training', 'Embedding images w/ resnet', 'Embedding images w/ resnet (autoencoder']
):
	analyze(model, dataset, desc)
