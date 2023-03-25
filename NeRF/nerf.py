#!/usr/bin/env python

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from helper_functions import *
from Network import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# code referred from tinyNerf pytorch implementation on GoogleColab
def get_rays(h,w,f,P):
    """
    Inputs: h: the height of the image in pixels. 
            w: The width of the image in pixels.
            f: The focal length of the image in pixel.
            P: The Projection matrix from camera to world coordinates system.

    Outputs:Origin_of_ray: The starting of the ray passing through the pixel in the image plane to the corresponding point in the world plane. 
            Direction_of_ray: The Direction of the ray passing through the pixel in the image plane to the corresponding point in the world plane.
    """
    

    R = P[:3,:3]
    i,j = get_meshgrid(torch.arange(w).to(P),
                       torch.arange(h).to(P))
    
    #The division by the focal length is necessary to convert the x and y pixel coordinates of the rays
    #in the image plane to the corresponding coordinates in camera coordinates.
    #The directions tensor is created by stacking three tensors of shape (height, width) representing the x, y, and z components of the direction of each ray in camera coordinates. The x and y components are computed by subtracting half the width and height of the image from ii and jj respectively, and then dividing by the focal length. The z component is set to -1
    x = (i-w* 0.5)/f
    y = -(j-h * 0.5)/f
    directions = torch.stack([x, y, -torch.ones_like(i)],dim = -1)
    # print("directions shape", directions.size())
    # print("P size", P[:3,:3].size())
    
    
    
    # The [..., None, :] notation is called "ellipsis" and it represents
    # all the dimensions in the tensor that are not explicitly mentioned in the indexing operation.
    # In this case, it represents the two dimensions height and width in directions. 
    # The None index adds a new dimension to the tensor, making it of shape [height, width, 1, 3]. 
    # The resulting tensor can be broadcast with the shape [4, 4] of the P[:3, :3] matrix in the following multiplication operation.
    Directions = directions[..., None, :]
    Direction_of_ray = torch.sum( Directions * R, dim= -1)
    # Direction_of_ray = torch.sum(directions[..., None, :] * P[:3,:3].unsqueeze(0).expand(directions.shape[0], directions.shape[1], -1, -1), dim= -1)
    # Direction_of_ray = torch.sum(directions[..., None, :] * P[:3,:3].unsqueeze(0).unsqueeze(0).expand(directions.shape[0], directions.shape[1], -1, -1), dim= -1)




    #The ray_origins tensor is obtained by repeating the translation component of the P matrix 
    #along the first two dimensions of the ray_directions tensor.
    Origin_of_ray = P[:3,-1].expand(Direction_of_ray.shape)
    # print(f"origin of ray dim:{Origin_of_ray.shape}")
    # print(f"origin of ray dim:{Direction_of_ray.shape}")


    return Origin_of_ray, Direction_of_ray


def positional_encoding(points, num_higher_freq = 6):
    """
    Inputs: points : sample points for the encoding to higher dimensions.
            num_higher_freq : how many higher dimension needed to be encoded.

    Outputs: high_dim_encoded_points: higher dimension encoded sample points.

    """
    encoding = [points]
    frequency_bands = 2.0 * torch.linspace(0.0,5.0,6,dtype=points.dtype,device=points.device, )
    print(f"frequency{frequency_bands}")


    for fequency in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(points * fequency))
    # print(len(encoding))
    encoding_points = torch.cat(encoding, dim=-1)
    # high_dim_encoded_points = [points]
    

    # for  i in range(num_higher_freq):
        
    #     high_dim_encoded_points.append(torch.sin(3*i*(points)))
    #     high_dim_encoded_points.append(torch.cos(3*i*(points)))
    # high_dim_encoded_points = torch.cat(high_dim_encoded_points, dim=-1)

    # return high_dim_encoded_points
    return encoding_points





def sampling_ray( Origin_of_ray, Direction_of_ray, threshold_near, threshold_far, sample_num, higher_frequencies = 6,noise = False): 
    """
    Inputs :


    
    Outputs: 
    
    
    """
    # print(f"orign shape {Origin_of_ray.shape}")
    depth_values = torch.linspace(threshold_near, threshold_far, sample_num)

    if noise == True:
        # s = list(Origin_of_ray.shape[:-1]) + [sample_num]
        s = [Origin_of_ray.shape[0], Origin_of_ray.shape[1], sample_num]
        # print(s)
        depth_values = depth_values  + torch.rand(s).to(Origin_of_ray) * (threshold_far-threshold_near) /sample_num
        # depth_values = depth_values + torch.rand(tuple(s)).to(Origin_of_ray) * (threshold_far-threshold_near) /sample_num
        # depth_values = depth_values + torch.rand(tuple(int(i) for i in s)).to(Origin_of_ray) * (threshold_far-threshold_near) /sample_num

        # noise_val = torch.tensor(np.random.random(size=sample_num) * (threshold_far - threshold_far)/sample_num)

        # noise_val = torch.tensor(np.random(size = sample_num) * (threshold_far - threshold_far)/sample_num)
        # depth_values = depth_values + noise_val
    # print(f"Origin_of_ray shape: {Origin_of_ray[...,None,:].shape}")
    # print(f"Direction_of_ray shape: {Direction_of_ray[...,None,:].shape}")
    # print(f"depth_values shape: {depth_values[...,:,None].shape}")
    rays = Origin_of_ray[...,None,:] + Direction_of_ray[...,None,:] * depth_values[...,:,None]
    
    # print(f"ray shape: {rays.shape}")
    sample_points = rays.reshape((-1,3))
    # print(f"sample points: {sample_points.shape}")
    sample_points = positional_encoding(sample_points, higher_frequencies)
    # print(f"sample points: {sample_points.shape}")
    return sample_points, depth_values, rays


def render_volume(rgb_values_, depth_values, direction_of_rays, Origin_of_rays):
    """
    Inputs: rgb_values : This are the network output rgb values at each sampling points
            depth_values : depth values of the sample 
            direction_of_rays : The Direction of ray  (H * W * 3)

    Outputs: Image with rgb along the direction specified

    """
    #as the rgb value cannot be negative, if any negative value identified making it zero
    # print(f"ray direction size{direction_of_rays.shape}")
    # print(f"egb_valuesshape{rgb_values_.shape}")
    rgb_values = torch.relu(rgb_values_[...,3])

    # taking the sigmoid of the rgb raw values to predict the kind of probability of the color
    rgb_actual = torch.special.expit(rgb_values_[...,:3])
    # print(f"depth_valuesjj{depth_values.shape}")
    #calculating the distance between two consecutive points between the ray
    delta_depth_values = depth_values[...,1:] - depth_values[...,:-1]
    # print(f"delta_depth_valuesja{delta_depth_values.shape}")
    diff_last_points = torch.tensor([1e12], dtype = Origin_of_rays.dtype, device = Origin_of_rays.device)
    diff_last_points = diff_last_points.expand(depth_values[...,:1].shape)
    delta_depth_values = torch.cat([delta_depth_values,diff_last_points], dim =-1)

    # direction_of_rays = direction_of_rays.reshape(rgb_values.shape[0], -1)

    # ray_direction = torch.norm(direction_of_rays[..., None, :], dim = -1)

    # delta_depth_values = delta_depth_values * ray_direction
    # print(f"the rgb_actual: {rgb_actual.shape}, the delta_depth : {delta_depth_values.shape}")
    # print(f"delta_depth_values{delta_depth_values.shape}")
    aplha = 1  - torch.exp(-rgb_values * delta_depth_values)


    transmittance = 1- aplha + 1e-10

    # diff_last_points = torch.cat([torch.ones((aplha.shape[0], 1)), transmittance],dim = -1)

    transmittance = get_cumulative_product(transmittance)

    weights = (aplha * transmittance).to(device)

    rgb_map = torch.sum(weights[...,None] * rgb_actual, -2)

    return rgb_map



#threshold_near,threshold_far, sample_num, higher_frequencies, noise = False
# def get_rendering(h,w,f,P,model,full_rendering, args):
#     """
    
#     """
#     Origin_of_ray, Direction_of_ray =  get_rays(h,w,f,P)

#     Direction_of_ray = Direction_of_ray.reshape((-1,3))
#     Origin_of_ray = Origin_of_ray.reshape((-1,3))

#     ray_samples = None
#     if full_rendering:
#         ray_samples = range(Direction_of_ray.shape[0])
#     else:
#         ray_samples = random.sample(range(Direction_of_ray.shape[0]), args.no_of_rays)
#     print(f"ray sample type{type(ray_samples)}")
#     Direction_of_ray = Direction_of_ray[ray_samples]
#     Origin_of_ray = Origin_of_ray[ray_samples]

#     sample_points, depth_values = sampling_ray(Origin_of_ray, Direction_of_ray, args.threshold_near, args.threshold_far, args.sample_num,higher_frequencies = 16,noise = True)
#     print(f"sample ppoints: {sample_points.shape} points")
#     print(f"data type of sample points: {sample_points.dtype}")

#     # sample_points = sample_points.reshape((-1, 3))

#     # encoded_points = positional_encoding(sample_points)
    
#     predicted_outputs = model(sample_points)
#     predicted_outputs = predicted_outputs.reshape((len(ray_samples), int(args.sample_num), 4))



#     rgb_prediction = render_volume(predicted_outputs, depth_values, Direction_of_ray)

#     return rgb_prediction, ray_samples


def get_rendering(h,w,f,P,model,full_rendering, args):
    """
    
    """
    Origin_of_ray, Direction_of_ray =  get_rays(h,w,f,P)

    # Direction_of_ray = Direction_of_ray.reshape((-1,3))
    # Origin_of_ray = Origin_of_ray.reshape((-1,3))

    sample_points, depth_values, rays = sampling_ray(Origin_of_ray, Direction_of_ray, args.threshold_near, args.threshold_far, args.sample_num,higher_frequencies = 16,noise = True)
    # print(f"sample ppoints: {sample_points.shape} points")
    depth_values = depth_values.squeeze()
    # print(f"depth_values: {depth_values.shape} points")
    sample_points = sample_points.to(torch.float64)
    # print(f"sample_points: {sample_points.shape} points")
    # print(f"ray ppoints: {rays.shape} points")/
    # print(f"ray ppointsshapw: {rays.shape[:-1]} points")
    # print(f"data type of sample points: {sample_points.dtype}")

    # sample_points = sample_points.reshape((-1, 3))

    # encoded_points = positional_encoding(sample_points)
    predicted_outputs = []
    batches = mini_batches(sample_points, 1000)
    # print(len(batches))
    for batch in batches:
        
        predicted_outputs.append(model(batch))
        # print(" yus wokring for current batch")
    # print("naccho 1 done")
    # print(f"predicted output shape:{len(predicted_outputs)}")


    radiance_field_flat = torch.cat(predicted_outputs, dim=0)
    # print(f"unflat_shape:{radiance_field_flat.shape}")
    unflat_shape = list(rays.shape[:-1]) + [4]
    # print(f"unflat_shape:{unflat_shape}")
    radiance_field = torch.reshape(radiance_field_flat, unflat_shape)

    # radiance_field_flat = torch.cat(predicted_outputs, dim=0)
    # flat_shape = [-1, radiance_field_flat.shape[-1]]
    # radiance_field = torch.reshape(radiance_field_flat, flat_shape)
    # radiance_field = torch.reshape(radiance_field, [1000, -1, 4])
    # radiance_field = radiance_field.reshape([100, 100, 6, 4])
    # predicted_outputs = predicted_outputs.reshape((len(args.ray_samples), int(args.sample_num), 4))



    rgb_prediction = render_volume(radiance_field, depth_values, Direction_of_ray, Origin_of_ray)

    return rgb_prediction








