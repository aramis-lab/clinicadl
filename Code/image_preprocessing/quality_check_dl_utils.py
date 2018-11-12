# -*- coding: utf-8 -*-


import sqlite3
import os
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from matplotlib.pyplot import imshow

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def load_nifti_images(image_path):

    image = nib.load(image_path)
    sample = np.array(image.get_data())

    # normalize input
    _min=np.min(sample)
    _max=np.max(sample)
    sample=(sample-_min)*(1.0/(_max-_min))-0.5
    sz=sample.shape
    input_images=[sample[:,:,int(sz[2]/2)],
                  sample[int(sz[0] / 2), :, :],
                  sample[:,int(sz[1]/2),:]]

    output_images=[np.zeros((224,224),),
                   np.zeros((224, 224)),
                   np.zeros((224, 224))]

    # flip, resize and crop
    for i in range(3):
        # try the dimension of input_image[i]
        # rotate the slice with 90 degree, I don't know why, but read from nifti file, the img has been rotated, thus we do not have the same direction with the pretrained model

        if len(input_images[i].shape) == 3:
            slice = np.reshape(input_images[i], (input_images[i].shape[0], input_images[i].shape[1]))
        else:
            slice = input_images[i]

        _scale=min(256.0/slice.shape[0],256.0/slice.shape[1])
        slice=transform.rescale(slice[::-1,:], _scale, mode='constant', clip=False)

        sz=slice.shape
        # pad image
        dummy=np.zeros((256,256),)
        dummy[int((256-sz[0])/2): int((256-sz[0])/2)+sz[0], int((256-sz[1])/2): int((256-sz[1])/2)+sz[1]] = slice

        #rotate and flip the iamge back to the right direction for each view, if the MRI was read by nibabel
        output_images[i] = np.flip(np.rot90(dummy[16:240, 16:240]), axis=1).copy() ## it seems that this will rotate the image 90 degree with counter-clockwise direction and then flip it horizontally

    return [torch.from_numpy(i).float().unsqueeze_(0) for i in output_images]
