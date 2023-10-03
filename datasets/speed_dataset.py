import os
import re
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

class SDDataset(Dataset):
	def __init__(self, data_dir, context_len=4):
		self._data_dir = data_dir
		self._df = pd.read_csv(f'{self._data_dir}/data.csv')
		imu_df = pd.read_csv(f'{self._data_dir}/imu_data.csv')
		self._df = self._df.merge(imu_df, on='frame')

		self._ctx_len = context_len

		regex = r'[^0-9]*(.*)_img.png'

		left_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_l')])
		right_files = set([int(re.search(regex, f).group(1)) for f in os.listdir(f'{self._data_dir}imgs_r')])

		self._df['in_files'] = False

		for f in left_files.intersection(right_files):
			self._df.loc[self._df['frame'] == f, 'in_files'] = True

		self._df = self._df[self._df.in_files]

	def __len__(self):
		return len(self._df) - self._ctx_len - 1

	def __getitem__(self, idx):
		start_idx = idx + 1
		end_idx = start_idx + self._ctx_len + 1

		sample_data = self._df.iloc[start_idx:end_idx]

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

		speed_x = torch.tensor(list(sample_data['vel_x']))
		speed_y = torch.tensor(list(sample_data['vel_y']))
		speed_z = torch.tensor(list(sample_data['vel_z']))

		acc_x = torch.tensor(list(sample_data['acc_x']))
		acc_y = torch.tensor(list(sample_data['acc_y']))
		acc_z = torch.tensor(list(sample_data['acc_z']))

		speed = torch.sqrt(speed_x ** 2 + speed_y ** 2 + speed_z ** 2) / 10

		imgs_l = torch.stack(imgs_left)
		imgs_r = torch.stack(imgs_right)

		acc_vector = torch.stack([acc_x, acc_y, acc_z], dim=1)

		return imgs_l.float()/255., imgs_r.float()/255., acc_vector, speed

if __name__ == '__main__':
	data = SDDataset('data/run_2023_08_24_17_10_06/', context_len=40)
