import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import os
import re

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

class LocationDataset(Dataset):
	def __init__(self, data_dir, context_len=4):
		self._data_dir = data_dir
		self._df = pd.read_csv(f'{self._data_dir}/data.csv')

		regex = r'[^0-9]*(.*)_img.png'

		left_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_l')])
		right_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_r')])

		self._df['in_files'] = False

		for f in left_files.intersection(right_files):
			self._df.loc[self._df['frame'] == f, 'in_files'] = True

		self._df = self._df[self._df.in_files]
		self._files = list(left_files.intersection(right_files)) + list(left_files.intersection(right_files))

	def __len__(self):
		return len(self._df)

	def __getitem__(self, idx):
		df_idx = idx if idx < (len(self._files) / 2) else int(idx / 2) + (idx % 2)

		try:
			sample_data = self._df.iloc[df_idx]
		except:
			print(idx)
			print(df_idx)
			print(len(self._df))
			raise

		if idx < (len(self._files) / 2):
			img = read_image(
				f'{self._data_dir}imgs_l/{int(sample_data.frame)}_img.png',
				mode=ImageReadMode.RGB
			)
		else:
			img = read_image(
				f'{self._data_dir}imgs_r/{int(sample_data.frame)}_img.png',
				mode=ImageReadMode.RGB
			)

		x = torch.tensor([sample_data['x']])
		y = torch.tensor([sample_data['y']])
		z = torch.tensor([sample_data['z']])

		return img / 255., torch.tensor([x, y, z])

if __name__ == '__main__':
	data = SDDataset('data/run_2023_08_24_17_10_06/', context_len=40)
