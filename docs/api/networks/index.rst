.. _api_networks:

:mod:`clinicadl.networks`
=========================

.. automodule:: clinicadl.networks

:mod:`clinicadl.networks.nn`
----------------------------

.. automodule:: clinicadl.networks.nn

.. currentmodule:: clinicadl.networks.nn

Build your own neural network
*****************************

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

   MLP
   ConvEncoder
   ConvDecoder
   CNN
   Generator
   AutoEncoder
   VAE

Common neural networks
**********************

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

   UNet
   AttentionUNet
   DenseNet
   ResNet
   SEResNet
   ViT

From literature
***************

.. autosummary::
   :nosignatures:
   :template: autosummary/function.rst

    get_densenet
    get_resnet
    get_seresnet
    get_vit

:mod:`clinicadl.networks.config`
--------------------------------

.. automodule:: clinicadl.networks.config

.. currentmodule:: clinicadl.networks.config

Build your own neural network
*****************************

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

   MLPConfig
   ConvEncoderConfig
   ConvDecoderConfig
   CNNConfig
   GeneratorConfig
   AutoEncoderConfig
   VAEConfig

Common neural networks
**********************

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

   UNetConfig
   AttentionUNetConfig
   DenseNetConfig
   ResNetConfig
   SEResNetConfig
   ViTConfig

From literature
***************

DenseNets
^^^^^^^^^

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

   DenseNet121Config
   DenseNet161Config
   DenseNet169Config
   DenseNet201Config

ResNets
^^^^^^^

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

   ResNet18Config
   ResNet34Config
   ResNet50Config
   ResNet101Config
   ResNet152Config

Squeeze-and-Excitation ResNets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

   SEResNet50Config
   SEResNet101Config
   SEResNet152Config

Vision Transformers
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

   ViTB16Config
   ViTB32Config
   ViTL16Config
   ViTL32Config

.. toctree::
   :hidden:

   nn
   config