import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import os
import re

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

class ImageDataset(Dataset):
	def __init__(self, data_dir):
		self._data_dir = data_dir
		self._df = pd.read_csv(f'{self._data_dir}/data.csv')
		left_files = set([f'{self._data_dir}imgs_l/{f}' for f in os.listdir(f'{self._data_dir}imgs_l')])
		right_files = set([f'{self._data_dir}imgs_r/{f}' for f in os.listdir(f'{self._data_dir}imgs_r')])

		self._files = list(left_files.union(right_files))

	def __len__(self):
		return len(self._files)

	def __getitem__(self, idx):
		img = read_image(self._files[idx], mode=ImageReadMode.RGB)

		return img.float() / 255., img.float() / 255.
