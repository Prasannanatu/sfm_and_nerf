#!/usr/bin/env python


from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from helper_functions import *
from Network import *
from nerf import *
import argparse
import tqdm as tq
import random
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import imageio

# Load images from the specified folder
image_folder = '/home/pvnatu/venv/bin/~venv/NeRF/NeRF/output/test_output'
images = []
for i in range(2, 180, 2):
    filename = f"test_{i}.jpg"
    images.append(imageio.imread(os.path.join(image_folder, filename)))

# Save GIF at the specified path
gif_path = '/home/pvnatu/venv/bin/~venv/NeRF/NeRF/output/test_output/test_gif.gif'
imageio.mimsave(gif_path, images, fps=60)