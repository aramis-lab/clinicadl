"""Config classes for :py:mod:`ClinicaDL neural networks <clinicadl.networks.nn>`."""

from ..nn.att_unet import AttentionUNetConfig
from ..nn.autoencoder import AutoEncoderConfig
from ..nn.cnn import CNNConfig
from ..nn.conv_decoder import ConvDecoderConfig
from ..nn.conv_encoder import ConvEncoderConfig
from ..nn.densenet import (
    DenseNet121Config,
    DenseNet161Config,
    DenseNet169Config,
    DenseNet201Config,
    DenseNetConfig,
)
from ..nn.generator import GeneratorConfig
from ..nn.mlp import MLPConfig
from ..nn.resnet import (
    ResNet18Config,
    ResNet34Config,
    ResNet50Config,
    ResNet101Config,
    ResNet152Config,
    ResNetConfig,
)
from ..nn.senet import (
    SEResNet50Config,
    SEResNet101Config,
    SEResNet152Config,
    SEResNetConfig,
)
from ..nn.unet import UNetConfig
from ..nn.utils.config import NetworkConfig
from ..nn.vae import VAEConfig
from ..nn.vit import ViTB16Config, ViTB32Config, ViTConfig, ViTL16Config, ViTL32Config
from .enum import ImplementedNetwork
