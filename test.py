import gc
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from unetplusplus import NestedUNet

from torch.utils.data.sampler import RandomSampler
from torchvision import transforms
from tqdm import tqdm
from dataloader import random_seed, PolypDataset
# from model import UNet

test_images_file = "data/test_images.txt"
test_labels_file = "data/test_masks.txt"

input_size = (128, 128)
torch.manual_seed(7)
test_set = PolypDataset(test_images_file, test_labels_file, input_size)
test_loader = DataLoader(test_set, batch_size = 1, sampler = RandomSampler(test_set))

# Inference device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
model_path = "experiment_test/polyp_unet1.pth"
# model = NestedUNet(n_channels = 3, n_classes = 1, bilinear = False).to(device)
model = NestedUNet(num_classes= 1, input_channels = 3, bilinear = False).to(device)

model.load_state_dict(torch.load(model_path, map_location = device))
model.eval()

