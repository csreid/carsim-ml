import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module, LSTM, Linear, Conv2d, functional as F
from torch.utils.data import DataLoader
from speed_dataset import SDDataset

class VisionInput(Module):
	def __init__(self, img_features, network='resnet'):
		super().__init__()
		self._img_features = img_features
		self._net = network

		if network != 'resnet':
			self.c1 = Conv2d(3, 8, (8,8), stride=4)
			self.c2 = Conv2d(8, 16, (6,6), stride=3)
			self.c3 = Conv2d(16, 32, (4,4), stride=2)
			self.c4 = Conv2d(32, 64, (2,2))

			self.fc = Linear(256, img_features)

		else:
			self.vision = resnet18(weights=ResNet18_Weights.DEFAULT)

			for param in self.vision.parameters():
					param.requires_grad = False

			self.vision.fc = Linear(512, self._img_features)

	def _forward(self, X):
		out = F.tanh(self.c1(X))
		out = F.tanh(self.c2(out))
		out = F.tanh(self.c3(out))
		out = F.tanh(self.c4(out))
		out = F.tanh(torch.flatten(out, start_dim=1))
		out = F.tanh(self.fc(out))

		return out

	def _resnet(self, X):
		return self.vision(X)

	def forward(self, X):
		if self._net == 'resnet':
			return self._resnet(X)

		return self._forward(X)

if __name__ == '__main__':
	vi = VisionInput(64)
	inp = torch.zeros(1, 3, 256, 256)
	print(vi(inp).shape)
