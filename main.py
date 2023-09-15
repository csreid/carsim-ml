import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
from speed_dataset import SDDataset
from torch.utils.data import DataLoader

data = SDDataset('data/run_2023_08_24_17_10_06/')
