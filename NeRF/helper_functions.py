#!/usr/bin/env python

from typing import Optional
import numpy as np
import os
import cv2
import glob
import json
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def get_meshgrid(tensor_value1, tensor_value2):
    """
    Inputs: get two tensor values
            tensor_value1: the width of the image plane in world.
            tensor_value2: the height of the image plane in world
    
    Outputs: tuple of tensor as which are transposed in the last two values of the meshgrid output
    
    """
    #Get the value required for the Tensor
    i, j  = torch.meshgrid(tensor_value1, tensor_value2) 

    #return the swapped last two dimension of the meshgrid output
    m = i.transpose(-1,-2)
    n = j.transpose(-1,-2)

    return m,n

def get_cumulative_product(Tense1):
    """
    Inputs: A tensor: for which cumulative product is necessary

    Outputs:Cumulative Product of the given tensor at a specified axis
    this is cumulative product equivalent of tensorflow function as it not exist in pytorch we are making the function
    thhat can give us the desired output.
    
    """
    #Pointing to last dimension of Tensor
    dim = -1

    #calculate the cumulative product for the given tensor at the specified dimension
    cumulative_product = torch.cumprod(Tense1, dim)

    #roll The dimennsion as we don't want the last elements to be a part of the cumulative product we roll so that
    #last elements become first and then changing all this first elements to be te 1
    cumulative_product = torch.roll(cumulative_product, 1, dim)
    cumulative_product[..., 0] = 1


    return cumulative_product


def load_model(model, args):
    start = 0

    checkpoint_files = glob.glob(os.path.join(args.checkpoint_path, '*.ckpt'))

    if not checkpoint_files:
        print("............No checkpoints found. Initiating New Model for training...........")

    else:
        
        latest_checkpoint_file = max(checkpoint_files, key = os.path.getctime)
        print(f"Found a latest checkpoint file: {latest_checkpoint_file}")


        if args.load_saved_checkpoint:
            checkpoint = torch.load(latest_checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            start = checkpoint['iteration'] + 1
            print(f"..........Loading the lastest points from : {latest_checkpoint_file}...............")

        else:
            print("...........initaiting new model with loading checkpoint_file........")


    return start


class NeRF_dataset(Dataset):
    """
    this is taken from the pytorch implementation of Data loader.
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self,root_dir, mode_type):
        """
        
        
        """
        self.tiny_data = np.load("tiny_nerf_data.npz")
        data_file = "transforms_train.json"
        self.root_dir = root_dir
        if mode_type  == "train":
            data_file = "transforms_train.json"
        if mode_type  == "test":
            data_file = "transforms_test.json"
        if mode_type  == "val":
            data_file = "transforms_val.json"

        path = os.path.join(root_dir, data_file)
        


        with open(path) as file:
            self.data = json.load(file)






    def __len__(self):
        return len(self.data["frames"])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.data["frames"][idx]["file_path"] + ".png"
        image_path = os.path.join(self.root_dir +os.sep + image)

        image = cv2.imread(image_path)

        image = image.resize(image, (250,250), interpolation= cv2.INTER_LANCZOS4)


        P = torch.tensor(self.data["frames"][idx]["transform_matrix"])



        horizontal_field_of_view = self.data["camera_angle_x"]

        focal_length = 0.5* image.shape[0]/math.tan(0.5*horizontal_field_of_view) # for getting focal length formula

        return focal_length, image, P
    
    def tiny_nerf_data(self, device):
        images = self.tiny_data["images"]
        images = torch.tensor(images).to(device)
        P = torch.tensor(self.tiny_data["poses"]).to(device)
        focal_length = torch.tensor(self.tiny_data["focal"])
        return focal_length, P, images


def mini_batches(inputs, batch_size):
    batch_size = 16384

    return [inputs[i:i + batch_size] for i in range(0, inputs.shape[0], batch_size)]
        
    
    
    

    











