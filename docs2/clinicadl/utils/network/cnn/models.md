# Models

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Cnn](./index.md#cnn) /
Models

> Auto-generated documentation for [clinicadl.utils.network.cnn.models](../../../../../clinicadl/utils/network/cnn/models.py) module.

- [Models](#models)
  - [Conv4_FC3](#conv4_fc3)
    - [Conv4_FC3.get_dimension](#conv4_fc3get_dimension)
    - [Conv4_FC3.get_input_size](#conv4_fc3get_input_size)
    - [Conv4_FC3.get_task](#conv4_fc3get_task)
  - [Conv5_FC3](#conv5_fc3)
    - [Conv5_FC3.get_dimension](#conv5_fc3get_dimension)
    - [Conv5_FC3.get_input_size](#conv5_fc3get_input_size)
    - [Conv5_FC3.get_task](#conv5_fc3get_task)
  - [ResNet3D](#resnet3d)
    - [ResNet3D.get_dimension](#resnet3dget_dimension)
    - [ResNet3D.get_input_size](#resnet3dget_input_size)
    - [ResNet3D.get_task](#resnet3dget_task)
  - [SqueezeExcitationCNN](#squeezeexcitationcnn)
    - [SqueezeExcitationCNN.get_dimension](#squeezeexcitationcnnget_dimension)
    - [SqueezeExcitationCNN.get_input_size](#squeezeexcitationcnnget_input_size)
    - [SqueezeExcitationCNN.get_task](#squeezeexcitationcnnget_task)
  - [Stride_Conv5_FC3](#stride_conv5_fc3)
    - [Stride_Conv5_FC3.get_dimension](#stride_conv5_fc3get_dimension)
    - [Stride_Conv5_FC3.get_input_size](#stride_conv5_fc3get_input_size)
    - [Stride_Conv5_FC3.get_task](#stride_conv5_fc3get_task)
  - [resnet18](#resnet18)
    - [resnet18.get_dimension](#resnet18get_dimension)
    - [resnet18.get_input_size](#resnet18get_input_size)
    - [resnet18.get_task](#resnet18get_task)
  - [get_layers_fn](#get_layers_fn)

## Conv4_FC3

[Show source in models.py:99](../../../../../clinicadl/utils/network/cnn/models.py#L99)

Convolutional neural network with 4 convolution and 3 fully-connected layer.
Reduce the 2D or 3D input image to an array of size output_size.

#### Signature

```python
class Conv4_FC3(CNN):
    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        ...
```

#### See also

- [CNN](../sub_network.md#cnn)

### Conv4_FC3.get_dimension

[Show source in models.py:163](../../../../../clinicadl/utils/network/cnn/models.py#L163)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### Conv4_FC3.get_input_size

[Show source in models.py:159](../../../../../clinicadl/utils/network/cnn/models.py#L159)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### Conv4_FC3.get_task

[Show source in models.py:167](../../../../../clinicadl/utils/network/cnn/models.py#L167)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## Conv5_FC3

[Show source in models.py:26](../../../../../clinicadl/utils/network/cnn/models.py#L26)

It is a convolutional neural network with 5 convolution and 3 fully-connected layer.
It reduces the 2D or 3D input image to an array of size output_size.

#### Signature

```python
class Conv5_FC3(CNN):
    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        ...
```

#### See also

- [CNN](../sub_network.md#cnn)

### Conv5_FC3.get_dimension

[Show source in models.py:90](../../../../../clinicadl/utils/network/cnn/models.py#L90)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### Conv5_FC3.get_input_size

[Show source in models.py:86](../../../../../clinicadl/utils/network/cnn/models.py#L86)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### Conv5_FC3.get_task

[Show source in models.py:94](../../../../../clinicadl/utils/network/cnn/models.py#L94)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## ResNet3D

[Show source in models.py:292](../../../../../clinicadl/utils/network/cnn/models.py#L292)

ResNet3D is a 3D neural network composed of 5 residual blocks. Each residual block
is compose of 3D convolutions followed by a batch normalization and an activation function.
It uses skip connections or shortcuts to jump over some layers. It's a 3D version of the
original implementation of Kaiming He et al.

Reference: Kaiming He et al., Deep Residual Learning for Image Recognition.
https://arxiv.org/abs/1512.03385?context=cs

#### Signature

```python
class ResNet3D(CNN):
    def __init__(
        self, input_size=[1, 169, 208, 179], gpu=False, output_size=2, dropout=0.5
    ):
        ...
```

#### See also

- [CNN](../sub_network.md#cnn)

### ResNet3D.get_dimension

[Show source in models.py:325](../../../../../clinicadl/utils/network/cnn/models.py#L325)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### ResNet3D.get_input_size

[Show source in models.py:321](../../../../../clinicadl/utils/network/cnn/models.py#L321)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### ResNet3D.get_task

[Show source in models.py:329](../../../../../clinicadl/utils/network/cnn/models.py#L329)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## SqueezeExcitationCNN

[Show source in models.py:334](../../../../../clinicadl/utils/network/cnn/models.py#L334)

SE-CNN is a combination of a ResNet-101 with Squeeze and Excitation blocks which was successfully
tested on brain tumour classification by Ghosal et al. 2019. SE blocks are composed of a squeeze
and an excitation step. The squeeze operation is obtained through an average pooling layer and
provides a global understanding of each channel.

The excitation part consists of a two-layer feed-forward network that outputs a vector of n values
corresponding to the weights of each channel of the feature maps.

Reference: Ghosal et al. Brain Tumor Classification Using ResNet-101 Based Squeeze and Excitation Deep Neural Network
https://ieeexplore.ieee.org/document/8882973

#### Signature

```python
class SqueezeExcitationCNN(CNN):
    def __init__(
        self, input_size=[1, 169, 208, 179], gpu=True, output_size=2, dropout=0.5
    ):
        ...
```

#### See also

- [CNN](../sub_network.md#cnn)

### SqueezeExcitationCNN.get_dimension

[Show source in models.py:371](../../../../../clinicadl/utils/network/cnn/models.py#L371)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### SqueezeExcitationCNN.get_input_size

[Show source in models.py:367](../../../../../clinicadl/utils/network/cnn/models.py#L367)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### SqueezeExcitationCNN.get_task

[Show source in models.py:375](../../../../../clinicadl/utils/network/cnn/models.py#L375)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## Stride_Conv5_FC3

[Show source in models.py:224](../../../../../clinicadl/utils/network/cnn/models.py#L224)

Convolutional neural network with 5 convolution and 3 fully-connected layer and a stride of 2 for each convolutional layer.
Reduce the 2D or 3D input image to an array of size output_size.

#### Signature

```python
class Stride_Conv5_FC3(CNN):
    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        ...
```

#### See also

- [CNN](../sub_network.md#cnn)

### Stride_Conv5_FC3.get_dimension

[Show source in models.py:283](../../../../../clinicadl/utils/network/cnn/models.py#L283)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### Stride_Conv5_FC3.get_input_size

[Show source in models.py:279](../../../../../clinicadl/utils/network/cnn/models.py#L279)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### Stride_Conv5_FC3.get_task

[Show source in models.py:287](../../../../../clinicadl/utils/network/cnn/models.py#L287)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## resnet18

[Show source in models.py:172](../../../../../clinicadl/utils/network/cnn/models.py#L172)

ResNet-18 is a neural network that is 18 layers deep based on residual block.
It uses skip connections or shortcuts to jump over some layers.
It is an image classification pre-trained model.
The model input has 3 channels in RGB order.

Reference: Kaiming He et al., Deep Residual Learning for Image Recognition.
https://arxiv.org/abs/1512.03385?context=cs

#### Signature

```python
class resnet18(CNN):
    def __init__(self, input_size, gpu=False, output_size=2, dropout=0.5):
        ...
```

#### See also

- [CNN](../sub_network.md#cnn)

### resnet18.get_dimension

[Show source in models.py:215](../../../../../clinicadl/utils/network/cnn/models.py#L215)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### resnet18.get_input_size

[Show source in models.py:211](../../../../../clinicadl/utils/network/cnn/models.py#L211)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### resnet18.get_task

[Show source in models.py:219](../../../../../clinicadl/utils/network/cnn/models.py#L219)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## get_layers_fn

[Show source in models.py:14](../../../../../clinicadl/utils/network/cnn/models.py#L14)

#### Signature

```python
def get_layers_fn(input_size):
    ...
```