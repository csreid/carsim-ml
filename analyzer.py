import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

import torch
from trainers.loc_prediction_from_imgs import analyze
from models.segmenter import Segmenter
from models.vision_input import VisionInput
from torch.nn import Sequential
from datasets.loc_est_dataset import LocationDataset

model1 = Sequential(
	VisionInput(2048, pretrained=False),
	Segmenter(2048),
)
model2 = Sequential(
	VisionInput(2048, pretrained=True),
	Segmenter(2048)
)
model3 = Sequential(
	VisionInput(2048),
	Segmenter(2048),
)
model3.load_state_dict(torch.load('foo.pt'))

dataset = LocationDataset('data/run_2023_09_18_20_49_33/')
analyze(model1, dataset, 'Before any training...')
analyze(model2, dataset, 'Pretrained ResNet')
analyze(model3, dataset, 'After training')
