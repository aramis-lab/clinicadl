"""
Copied from https://github.com/vfonov/deep-qc/blob/master/python/model/resnet_qc.py
"""

import torch.nn as nn
from torch.utils.data import Dataset
import nibabel as nib
from os import path
import torch

from ..tools.deep_learning.data import FILENAME_TYPE


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetQC(nn.Module):

    def __init__(self, block, layers, num_classes=2, use_ref=False, zero_init_residual=False):
        super(ResNetQC, self).__init__()
        self.inplanes = 64
        self.use_ref = use_ref
        self.feat = 3
        self.expansion = block.expansion
        self.conv1 = nn.Conv2d(2 if self.use_ref else 1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # for merging multiple features
        self.addon = nn.Sequential(
            nn.Conv2d(self.feat * 512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1, padding=0,
                      bias=True),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * block.expansion, 32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # split feats into batches
        x = x.view(-1, 2 if self.use_ref else 1, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # merge batches together
        x = x.view(-1, 512 * self.feat * self.expansion, 7, 7)
        x = self.addon(x)
        x = x.view(x.size(0), -1)

        return x


def resnet_qc_18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNetQC(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class QCDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, img_dir, data_df, use_extracted_tensors=False):
        """
        Args:
            img_dir (string): Directory of all the images.
            data_df (DataFrame): Subject and session list.

        """
        from ..tools.deep_learning.data import MinMaxNormalization

        self.img_dir = img_dir
        self.df = data_df
        self.use_extracted_tensors = use_extracted_tensors

        if ('session_id' not in list(self.df.columns.values)) or ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id']")

        self.normalization = MinMaxNormalization()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        subject = self.df.loc[idx, 'participant_id']
        session = self.df.loc[idx, 'session_id']

        if self.use_extracted_tensors:
            image_path = path.join(self.img_dir, 'subjects', subject, session, 'deeplearning_prepare_data',
                                   'image_based', 't1_linear',
                                   '%s_%s%s.pt' % (subject, session, FILENAME_TYPE["full"]))

            image = torch.load(image_path)
            image = self.pt_transform(image)
        else:
            image_path = path.join(self.img_dir, 'subjects', subject, session, 't1_linear',
                                   '%s_%s%s.nii.gz' % (subject, session, FILENAME_TYPE["full"]))

            image = nib.load(image_path)
            image = self.nii_transform(image)

        sample = {'image': image, 'participant_id': subject, 'session_id': session}

        return sample

    @staticmethod
    def nii_transform(image):
        import numpy as np
        import torch
        from skimage import transform

        sample = np.array(image.get_data())

        # normalize input
        _min = np.min(sample)
        _max = np.max(sample)
        sample = (sample - _min) * (1.0 / (_max - _min)) - 0.5
        sz = sample.shape
        input_images = [
            sample[:, :, int(sz[2] / 2)],
            sample[int(sz[0] / 2), :, :],
            sample[:, int(sz[1] / 2), :]
        ]

        output_images = [
            np.zeros((224, 224),),
            np.zeros((224, 224)),
            np.zeros((224, 224))
        ]

        # flip, resize and crop
        for i in range(3):
            # try the dimension of input_image[i]
            # rotate the slice with 90 degree, I don't know why, but read from
            # nifti file, the img has been rotated, thus we do not have the same
            # direction with the pretrained model

            if len(input_images[i].shape) == 3:
                slice = np.reshape(
                    input_images[i], (input_images[i].shape[0], input_images[i].shape[1]))
            else:
                slice = input_images[i]

            _scale = min(256.0 / slice.shape[0], 256.0 / slice.shape[1])
            # slice[::-1, :] is to flip the first axis of image
            slice = transform.rescale(
                slice[::-1, :], _scale, mode='constant', clip=False)

            sz = slice.shape
            # pad image
            dummy = np.zeros((256, 256),)
            dummy[int((256 - sz[0]) / 2): int((256 - sz[0]) / 2) + sz[0],
                  int((256 - sz[1]) / 2): int((256 - sz[1]) / 2) + sz[1]] = slice

            # rotate and flip the image back to the right direction for each view, if the MRI was read by nibabel
            # it seems that this will rotate the image 90 degree with
            # counter-clockwise direction and then flip it horizontally
            output_images[i] = np.flip(
                np.rot90(dummy[16:240, 16:240]), axis=1).copy()

        return torch.cat([torch.from_numpy(i).float().unsqueeze_(0) for i in output_images]).unsqueeze_(0)

    def pt_transform(self, image):
        from torch.nn.functional import interpolate, pad

        image = self.normalization(image) - 0.5
        image = image[0, :, :, :]
        sz = image.shape
        input_images = [
            image[:, :, int(sz[2] / 2)],
            image[int(sz[0] / 2), :, :],
            image[:, int(sz[1] / 2), :]
        ]

        output_images = list()

        # flip, resize and crop
        for slice in input_images:

            scale = min(256.0 / slice.shape[0], 256.0 / slice.shape[1])
            # slice[::-1, :] is to flip the first axis of image
            slice = interpolate(torch.flip(slice, (0,)).unsqueeze(0).unsqueeze(0), scale_factor=scale)
            slice = slice[0, 0, :, :]

            padding = self.get_padding(slice)
            slice = pad(slice, padding)

            # rotate and flip the image back to the right direction for each view, if the MRI was read by nibabel
            # it seems that this will rotate the image 90 degree with
            # counter-clockwise direction and then flip it horizontally
            output_images.append(torch.flip(torch.rot90(slice[16:240, 16:240], 1, [0, 1]), [1, ]).clone())

        return torch.cat([image.float().unsqueeze_(0) for image in output_images]).unsqueeze_(0)

    @staticmethod
    def get_padding(image):
        max_w = 256
        max_h = 256

        imsize = image.shape
        h_padding = (max_w - imsize[1]) / 2
        v_padding = (max_h - imsize[0]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

        padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))

        return padding
