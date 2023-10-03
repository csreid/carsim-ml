import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module, LSTM, Linear, ConvTranspose2d, functional as F
from torch.utils.data import DataLoader
from models.vision_input import VisionInput

class SpeedModel(Module):
	def __init__(self, img_features, pretrained=True, dev='cpu'):
		super().__init__()
		self._img_features = img_features
		self.vision = VisionInput(self._img_features, pretrained)

		self.precurrent1 = Linear(self._img_features*2, 256)
		self.precurrent2 = Linear(256, 256)
		self.speed_recurrent = LSTM(259, 64, batch_first=True)
		self.speed_out = Linear(64, 1)

		self.acc_feats = Linear(3, 3)
		self._device = dev

	def forward(self, imgs_l, imgs_r, acc_vectors):
		# batch, seq, channels, x, y
		batch_size = imgs_l.shape[0]
		seq_len = imgs_l.shape[1]
		channels = imgs_l.shape[2]
		x_pixels = imgs_l.shape[3]
		y_pixels = imgs_l.shape[4]

		img_feats_l = []
		for idx in range(seq_len):
			img_feats_l.append(self.vision(imgs_l[:, idx, :, :, :]))
		img_feats_l = torch.stack(img_feats_l, dim=1)

		img_feats_r = []
		for idx in range(seq_len):
			img_feats_r.append(self.vision(imgs_r[:, idx, :, :, :]))
		img_feats_r = torch.stack(img_feats_r, dim=1)

		out = torch.cat([img_feats_r, img_feats_l], dim=2)
		out = F.tanh(self.precurrent1(out))
		out = F.tanh(self.precurrent2(out))

		acc = F.tanh(self.acc_feats(acc_vectors))
		out = torch.cat([out, acc], dim=2)
		out, _ = self.speed_recurrent(out)
		out = self.speed_out(out).squeeze(2)
		out = F.tanh(out)

		return out

	def to(self, device):
		new = super().to(device)
		self._device = device

		return new

if __name__ == '__main__':
	v = Vision()
	data = SDDataset('data/run_2023_08_24_17_10_06/')
	loader = DataLoader(data, batch_size=4, shuffle=True)
	imgs_l, imgs_r, speeds = next(iter(loader))

	print(v(imgs_l.float(), imgs_r.float()).shape)
