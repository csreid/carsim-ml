import time
import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import MSELoss, HuberLoss, Sequential, Linear
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def analyze(model, dataset, desc):
	with torch.no_grad():
		model = model.to('cuda:0')
		loader = DataLoader(dataset, batch_size=64, shuffle=True)
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

		train_embs = all_embs[:15000]
		train_pts = pts[:15000]

		test_embs = all_embs[15000:]
		test_pts = pts[15000:]

		reg = LinearRegression()
		reg.fit(train_embs, train_pts)

		preds = reg.predict(test_embs)
		r2_x = r2_score(preds[:, 0], test_pts[:, 0])
		r2_y = r2_score(preds[:, 1], test_pts[:, 1])
		tqdm.write(f'R2 value (X-coordinate): {r2_x}')
		tqdm.write(f'R2 value (Y-coordinate): {r2_y}')

		dists = np.diagonal(euclidean_distances(preds, test_pts))
		tqdm.write(f'Mean distance: {np.mean(dists)}')
		tqdm.write(f'Distance stddev: {np.std(dists)}')

		fig, (ax1, ax2) = plt.subplots(ncols=2)
		ax1.plot(reg.predict(test_embs)[:400, 0], test_pts[:400, 0], 'ro', alpha=0.2)
		ax1.set_ylabel('True X-value')
		ax1.set_xlabel('X-value predicted from LR')

		ax2.plot(reg.predict(test_embs)[:400, 1], test_pts[:400, 1], 'ro', alpha=0.2)
		ax2.set_ylabel('True Y-value')
		ax2.set_xlabel('Y-value predicted from LR')
		plt.show()
