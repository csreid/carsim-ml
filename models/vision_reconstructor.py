import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module, LSTM, Linear, ConvTranspose2d, functional as F, Sequential, ReLU, Sigmoid

class VisionReconstructor(Module):
	def __init__(self, img_features):
		super().__init__()

		self.fc = Linear(img_features, 4096)
		self.deconv = Sequential(
			ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			Sigmoid(),
			ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
			Sigmoid(),
			ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
			Simgoid(),
			ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			Simgoid(),
			ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			Simgoid(),
			ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
		)

	def forward(self, X):
		out = F.tanh(self.fc(X))
		out = out.reshape(-1, 256, 4, 4)
		out = self.deconv(out)

		return out
