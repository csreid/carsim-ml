import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import os
import re

import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

class SegmentationDataset(Dataset):
	def __init__(self, data_dir, total=None):
		self._data_dir = data_dir
		self._df = pd.read_csv(f'{self._data_dir}/data.csv')

		regex = r'[^0-9]*(.*)_img.png'

		left_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_l')])
		right_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_r')])
		left_seg_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_seg_l')])
		right_seg_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_seg_r')])

		self._df['in_files'] = False

		for f in left_files.intersection(right_files):
			self._df.loc[self._df['frame'] == f, 'in_files'] = True

		self._df = self._df[self._df.in_files]
		self._files = list(left_files.intersection(right_files).intersection(left_seg_files).intersection(right_seg_files))
		if total:
			random.shuffle(self._files)
			self._files = self._files[:total]

	def __len__(self):
		return len(self._files)

	def __getitem__(self, idx):
		file = self._files[idx]
		if idx < (len(self._files) / 2):
			img = read_image(
				f'{self._data_dir}imgs_l/{file}_img.png',
				mode=ImageReadMode.RGB
			)
			seg_img = read_image(
				f'{self._data_dir}imgs_seg_l/{file}_img.png',
				mode=ImageReadMode.RGB
			)
		else:
			img = read_image(
				f'{self._data_dir}imgs_r/{file}_img.png',
				mode=ImageReadMode.RGB
			)
			seg_img = read_image(
				f'{self._data_dir}imgs_seg_r/{file}_img.png',
				mode=ImageReadMode.RGB
			)

		return img / 255., seg_img[0, :].long()
