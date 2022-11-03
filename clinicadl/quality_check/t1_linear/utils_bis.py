"""
Copied from https://github.com/vfonov/deep-qc/blob/master/python/model/resnet_qc.py
"""

from os import path

import nibabel as nib
import torch
import torch.nn as nn
from clinica.utils.input_files import T1W_LINEAR_CROPPED
from clinica.utils.inputs import clinica_file_reader
from torch.utils.data import Dataset
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from clinicadl.utils.caps_dataset.data import CapsDatasetImage
from torchvision import models
from torch.nn.parameter import Parameter

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor ) -> Tensor :
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

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetQC(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 2,
	    use_ref:bool = False,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetQC, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.use_ref = use_ref
        self.feat = 3
        self.inplanes = 64
        self.dilation = 1
        self.expansion = block.expansion

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(2 if self.use_ref else 1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for merging multiple features
        self.addon = nn.Sequential(
            nn.Conv2d(self.feat * 512 * block.expansion, 512*block.expansion, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512*block.expansion),
            nn.ReLU(inplace=False),
            nn.Conv2d(512*block.expansion, 32, kernel_size=1, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5,inplace=False),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        # split feats into batches
        # for core resnet based part
        x = x.view(-1, 2 if self.use_ref else 1 ,224,224)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # merge batches together
        x = x.view(-1, 512*self.feat * self.expansion, 7, 7)
        x = self.addon(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


    def load_from_std(self, std_model: models.ResNet ):
        # import weights from the standard ResNet model
        # TODO: finish
        # first load all standard items
        own_state = self.state_dict()
        for name, param in std_model.state_dict().items():
            if name == 'conv1.weight':
                if isinstance(param, Parameter):
                    param = param.data
                # convert to mono weight
                # collaps parameters along second dimension, emulating grayscale feature 
                mono_param=param.sum( 1, keepdim=True )
                if self.use_ref:
                    own_state[name].copy_( torch.cat((mono_param,mono_param),1) )
                else:
                    own_state[name].copy_( mono_param )
                pass
            elif name == 'fc.weight' or name == 'fc.bias' or name == 'conv2.weight' or name == 'conv2.bias':
                # don't use at all
                pass
            elif name in own_state:
                if isinstance(param, Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            


def _resnet_qc(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    progress: bool,
    **kwargs: Any
) -> ResNetQC:
    return ResNetQC(block, layers, **kwargs)


def resnet_qc_18(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNetQC:
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet_qc( BasicBlock, [2, 2, 2, 2], progress,
                   **kwargs)
    if pretrained:
        # load basic Resnet model
        model_ft = models.resnet18(pretrained=True)
        model.load_from_std(model_ft)
    return model

def resnet_qc_34(pretrained=False,progress: bool = True, **kwargs) -> ResNetQC:
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet_qc(BasicBlock, [3, 4, 6, 3], progress,
                   **kwargs)
    if pretrained:
        # load basic Resnet model
        model_ft = models.resnet34(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_50(pretrained: bool=False,progress: bool = True, **kwargs) -> ResNetQC:
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet_qc(Bottleneck, [3, 4, 6, 3], progress, **kwargs)

    if pretrained:
        model_ft = models.resnet50(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_101(pretrained: bool=False,progress: bool = True, **kwargs) -> ResNetQC:
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet_qc( Bottleneck, [3, 4, 23, 3], progress,
                   **kwargs)
    if pretrained:
        model_ft = models.resnet101(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_152(pretrained: bool=False, progress: bool = True, **kwargs)  -> ResNetQC:
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet_qc( Bottleneck, [3, 8, 36, 3], progress, **kwargs)

    if pretrained:
        model_ft = models.resnet152(pretrained=True,progress=progress)
        model.load_from_std(model_ft)
    return model


def resnext_qc_50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetQC:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = _resnet_qc(Bottleneck, [3, 4, 6, 3], progress, **kwargs)
    if pretrained:
        model_ft = models.resnext_50_32x4d(pretrained=True, progress=progress)
        model.load_from_std(model_ft)
    return model


def resnext_qc_101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetQC:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model =  _resnet_qc(Bottleneck, [3, 4, 23, 3], progress, **kwargs)

    if pretrained:
        model_ft = models.resnext_32x8d(pretrained=True, progress=progress)
        model.load_from_std(model_ft)
    return model


def wide_resnet_qc_50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetQC:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    model = _resnet_qc(Bottleneck, [3, 4, 6, 3], progress, **kwargs)

    if pretrained:
        model_ft = models.wide_resnet_50_2(pretrained=True, progress=progress)
        model.load_from_std(model_ft)
    return model


def wide_resnet_qc_101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetQC:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    model = _resnet_qc(Bottleneck, [3, 4, 23, 3], progress, **kwargs)
    if pretrained:
        model_ft = models.wide_resnet_101_2(pretrained=True, progress=progress)
        model.load_from_std(model_ft)
    return model

    
class QCDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, img_dir, data_df, use_extracted_tensors=True):
        """
        Args:
            img_dir (string): Directory of all the images.
            data_df (DataFrame): Subject and session list.

        """
        from clinicadl.utils.caps_dataset.data import MinMaxNormalization

        self.img_dir = img_dir
        self.df = data_df
        self.use_extracted_tensors = use_extracted_tensors

        if ("session_id" not in list(self.df.columns.values)) or (
            "participant_id" not in list(self.df.columns.values)
        ):
            raise Exception(
                "the data file is not in the correct format."
                "Columns should include ['participant_id', 'session_id']"
            )

        self.normalization = MinMaxNormalization()
        preprocessing_dict = {
            "preprocessing": "t1-linear",
            "mode": "image",
            "use_uncropped_image": False,
            "file_type": T1W_LINEAR_CROPPED,
        }
        self.tensor_dataset = CapsDatasetImage(
            img_dir,
            data_df,
            preprocessing_dict,
            label_presence=False,
            all_transformations=MinMaxNormalization(),
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.use_extracted_tensors:
            image = self.tensor_dataset[idx]
            image = self.pt_transform(image)
        else:
            subject = self.df.loc[idx, "participant_id"]
            session = self.df.loc[idx, "session_id"]
            image_path = clinica_file_reader(
                [subject], [session], self.img_dir, T1W_LINEAR_CROPPED
            )[0]
            image = nib.load(image_path[0])
            image = self.nii_transform(image)

        sample = {"image": image, "participant_id": subject, "session_id": session}

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
            sample[:, int(sz[1] / 2), :],
        ]

        output_images = [
            np.zeros(
                (224, 224),
            ),
            np.zeros((224, 224)),
            np.zeros((224, 224)),
        ]

        # flip, resize and crop
        for i in range(3):
            # try the dimension of input_image[i]
            # rotate the slice with 90 degree, I don't know why, but read from
            # nifti file, the img has been rotated, thus we do not have the same
            # direction with the pretrained model

            if len(input_images[i].shape) == 3:
                slice = np.reshape(
                    input_images[i],
                    (input_images[i].shape[0], input_images[i].shape[1]),
                )
            else:
                slice = input_images[i]

            _scale = min(256.0 / slice.shape[0], 256.0 / slice.shape[1])
            # slice[::-1, :] is to flip the first axis of image
            slice = transform.rescale(
                slice[::-1, :], _scale, mode="constant", clip=False
            )

            sz = slice.shape
            # pad image
            dummy = np.zeros(
                (256, 256),
            )
            dummy[
                int((256 - sz[0]) / 2) : int((256 - sz[0]) / 2) + sz[0],
                int((256 - sz[1]) / 2) : int((256 - sz[1]) / 2) + sz[1],
            ] = slice

            # rotate and flip the image back to the right direction for each view, if the MRI was read by nibabel
            # it seems that this will rotate the image 90 degree with
            # counter-clockwise direction and then flip it horizontally
            output_images[i] = np.flip(np.rot90(dummy[16:240, 16:240]), axis=1).copy()

        return torch.cat(
            [torch.from_numpy(i).float().unsqueeze_(0) for i in output_images]
        ).unsqueeze_(0)

    def pt_transform(self, image):
        from torch.nn.functional import interpolate, pad

        image = self.normalization(image) - 0.5
        image = image[0, :, :, :]
        sz = image.shape
        input_images = [
            image[:, :, int(sz[2] / 2)],
            image[int(sz[0] / 2), :, :],
            image[:, int(sz[1] / 2), :],
        ]

        output_images = list()

        # flip, resize and crop
        for slice in input_images:

            scale = min(256.0 / slice.shape[0], 256.0 / slice.shape[1])
            # slice[::-1, :] is to flip the first axis of image
            slice = interpolate(
                torch.flip(slice, (0,)).unsqueeze(0).unsqueeze(0), scale_factor=scale
            )
            slice = slice[0, 0, :, :]

            padding = self.get_padding(slice)
            slice = pad(slice, padding)

            # rotate and flip the image back to the right direction for each view, if the MRI was read by nibabel
            # it seems that this will rotate the image 90 degree with
            # counter-clockwise direction and then flip it horizontally
            output_images.append(
                torch.flip(
                    torch.rot90(slice[16:240, 16:240], 1, [0, 1]),
                    [
                        1,
                    ],
                ).clone()
            )

        return torch.cat(
            [image.float().unsqueeze_(0) for image in output_images]
        ).unsqueeze_(0)

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
