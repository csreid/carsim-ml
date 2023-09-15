import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import os
import re

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

class SDDataset(Dataset):
	def __init__(self, data_dir, context_len=4):
		self._data_dir = data_dir
		self._df = pd.read_csv(f'{self._data_dir}/data.csv')
		self._ctx_len = context_len

		regex = r'[^0-9]*(.*)_img.png'

		left_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_l')])
		right_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_r')])

		self._df['in_files'] = False

		for f in left_files.intersection(right_files):
			self._df.loc[self._df['frame'] == f, 'in_files'] = True

		self._df = self._df[self._df.in_files]

		print(self._df.head())

	def __len__(self):
		return len(self._df) - self._ctx_len - 1

	def __getitem__(self, idx):
		start_idx = idx + 1
		end_idx = start_idx + self._ctx_len + 1

		sample_data = self._df.iloc[start_idx:end_idx]
		print(sample_data)

		imgs_left = [
			read_image(
				f'{self._data_dir}imgs_l/{int(d.frame)}_img.png',
				mode=ImageReadMode.RGB
			)
			for _, d in sample_data.iterrows()
		]
		imgs_right = [
			read_image(
				f'{self._data_dir}imgs_r/{int(d.frame)}_img.png',
				mode=ImageReadMode.RGB
			)
			for _, d in sample_data.iterrows()
		]

		imgs_l = torch.stack(imgs_left)
		imgs_r = torch.stack(imgs_right)

		return imgs_l, imgs_r

if __name__ == '__main__':
	ds = SDDataset('data/run_2023_08_24_17_10_06/')

	print(ds.__getitem__(0))
