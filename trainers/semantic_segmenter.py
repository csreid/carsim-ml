import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module, LSTM, Linear, ConvTranspose2d, functional as F
from torch.utils.data import DataLoader
from speed_dataset import SDDataset

class VisionReconstructor(Module):
	def __init__(self, img_features):
		super().__init__()

		self.fc = Linear(img_features, 1024)
		self.t1 = ConvTranspose2d(64, 32, kernel_size=(2,2))
		self.t2 = ConvTranspose2d(32, 16, kernel_size=(4,4), stride=2)
		self.t3 = ConvTranspose2d(16, 8, kernel_size=(6,6), stride=3)
		self.t4 = ConvTranspose2d(8, 12, kernel_size=(8,8), stride=4)

	def forward(self, X):
		out = self.fc(X)
		out = out.reshape(-1, 64, 4, 4)
		out = F.sigmoid(self.t1(out))
		out = F.sigmoid(self.t2(out))
		out = F.sigmoid(self.t3(out))
		out = F.sigmoid(self.t4(out))[:, :, :128, :128]

		return out

if __name__ == '__main__':
	print(f'Output shape: {VisionReconstructor(64)._output_shape(4)}')
