import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module, LSTM, Linear, Conv2d, functional as F
from torch.utils.data import DataLoader

class VisionInput(Module):
	def __init__(self, img_features, pretrained=False):
		super().__init__()
		self._img_features = img_features

		if pretrained:
			tqdm.write(f'Using pretrained ResNet weights')
			self.vision = resnet18(weights=ResNet18_Weights.DEFAULT)
			for param in self.vision.parameters():
					param.requires_grad = False
		else:
			tqdm.write(f'Using fresh weights for ResNet')
			self.vision = resnet18()

		self.vision.fc = Linear(512, self._img_features)

	def forward(self, X):
		return self.vision(X)
