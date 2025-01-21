import torch
import tqdm
from visualization import plot_all_poses
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


with np.load('loftr_parameters.npz') as data:
    theta = data['theta']
    trans = data['trans']
    key3d = data['key3d']
key3d[:,:2]
key3d[:,2]= 1
dtype = torch.float32
theta = torch.tensor(theta).to(dtype)
trans = torch.tensor(trans).to(dtype)
theta = torch.inverse(theta)
trans = (theta @ -trans.unsqueeze(-1) ).squeeze(-1)
poses = torch.cat([theta, trans.unsqueeze(-1)], dim=-1) 
plot_all_poses(poses.numpy(), key3d)
