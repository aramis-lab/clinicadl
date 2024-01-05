# Models

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Quality Check](../index.md#quality-check) /
[T1 Linear](./index.md#t1-linear) /
Models

> Auto-generated documentation for [clinicadl.quality_check.t1_linear.models](../../../../clinicadl/quality_check/t1_linear/models.py) module.

- [Models](#models)
  - [BasicBlock](#basicblock)
    - [BasicBlock().forward](#basicblock()forward)
  - [Bottleneck](#bottleneck)
    - [Bottleneck().forward](#bottleneck()forward)
  - [Fire](#fire)
    - [Fire().forward](#fire()forward)
  - [ResNet_DarqQC](#resnet_darqqc)
    - [ResNet_DarqQC().forward](#resnet_darqqc()forward)
    - [ResNet_DarqQC().load_from_std](#resnet_darqqc()load_from_std)
  - [ResNet_DeepQC](#resnet_deepqc)
    - [ResNet_DeepQC().forward](#resnet_deepqc()forward)
  - [SqueezeNetQC](#squeezenetqc)
    - [SqueezeNetQC().forward](#squeezenetqc()forward)
    - [SqueezeNetQC().load_from_std](#squeezenetqc()load_from_std)
  - [conv1x1](#conv1x1)
  - [conv3x3](#conv3x3)
  - [resnet_darq_qc_18](#resnet_darq_qc_18)
  - [resnet_deep_qc_18](#resnet_deep_qc_18)
  - [squeezenet_qc](#squeezenet_qc)

## BasicBlock

[Show source in models.py:33](../../../../clinicadl/quality_check/t1_linear/models.py#L33)

#### Signature

```python
class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        ...
```

### BasicBlock().forward

[Show source in models.py:63](../../../../clinicadl/quality_check/t1_linear/models.py#L63)

#### Signature

```python
def forward(self, x: Tensor) -> Tensor:
    ...
```



## Bottleneck

[Show source in models.py:82](../../../../clinicadl/quality_check/t1_linear/models.py#L82)

#### Signature

```python
class Bottleneck(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        ...
```

### Bottleneck().forward

[Show source in models.py:117](../../../../clinicadl/quality_check/t1_linear/models.py#L117)

#### Signature

```python
def forward(self, x: Tensor) -> Tensor:
    ...
```



## Fire

[Show source in models.py:493](../../../../clinicadl/quality_check/t1_linear/models.py#L493)

#### Signature

```python
class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        ...
```

### Fire().forward

[Show source in models.py:506](../../../../clinicadl/quality_check/t1_linear/models.py#L506)

#### Signature

```python
def forward(self, x):
    ...
```



## ResNet_DarqQC

[Show source in models.py:140](../../../../clinicadl/quality_check/t1_linear/models.py#L140)

resnet copied from https://github.com/vfonov/darq/blob/main/python/model/resnet_qc.py

#### Signature

```python
class ResNet_DarqQC(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 2,
        use_ref: bool = False,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        ...
```

#### See also

- [BasicBlock](#basicblock)
- [Bottleneck](#bottleneck)

### ResNet_DarqQC().forward

[Show source in models.py:318](../../../../clinicadl/quality_check/t1_linear/models.py#L318)

#### Signature

```python
def forward(self, x: Tensor) -> Tensor:
    ...
```

### ResNet_DarqQC().load_from_std

[Show source in models.py:321](../../../../clinicadl/quality_check/t1_linear/models.py#L321)

#### Signature

```python
def load_from_std(self, std_model: models.ResNet):
    ...
```



## ResNet_DeepQC

[Show source in models.py:386](../../../../clinicadl/quality_check/t1_linear/models.py#L386)

renet copied from https://github.com/vfonov/deep-qc/blob/master/python/model/resnet_qc.py

#### Signature

```python
class ResNet_DeepQC(nn.Module):
    def __init__(
        self, block, layers, num_classes=2, use_ref=False, zero_init_residual=False
    ):
        ...
```

### ResNet_DeepQC().forward

[Show source in models.py:466](../../../../clinicadl/quality_check/t1_linear/models.py#L466)

#### Signature

```python
def forward(self, x):
    ...
```



## SqueezeNetQC

[Show source in models.py:517](../../../../clinicadl/quality_check/t1_linear/models.py#L517)

squezenet copied from https://github.com/vfonov/DARQ/blob/main/python/model/squezenet_qc.py

#### Signature

```python
class SqueezeNetQC(nn.Module):
    def __init__(self, version=1.0, num_classes=2, use_ref=False):
        ...
```

### SqueezeNetQC().forward

[Show source in models.py:582](../../../../clinicadl/quality_check/t1_linear/models.py#L582)

#### Signature

```python
def forward(self, x):
    ...
```

### SqueezeNetQC().load_from_std

[Show source in models.py:592](../../../../clinicadl/quality_check/t1_linear/models.py#L592)

#### Signature

```python
def load_from_std(self, std_model):
    ...
```



## conv1x1

[Show source in models.py:28](../../../../clinicadl/quality_check/t1_linear/models.py#L28)

1x1 convolution

#### Signature

```python
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    ...
```



## conv3x3

[Show source in models.py:12](../../../../clinicadl/quality_check/t1_linear/models.py#L12)

3x3 convolution with padding

#### Signature

```python
def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    ...
```



## resnet_darq_qc_18

[Show source in models.py:370](../../../../clinicadl/quality_check/t1_linear/models.py#L370)

Constructs a ResNet-18 model.

#### Arguments

- `pretrained` *bool* - If True, returns a model pre-trained on ImageNet
- `progress` *bool* - If True, displays a progress bar of the download to stderr

#### Signature

```python
def resnet_darq_qc_18(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> ResNet_DarqQC:
    ...
```

#### See also

- [ResNet_DarqQC](#resnet_darqqc)



## resnet_deep_qc_18

[Show source in models.py:487](../../../../clinicadl/quality_check/t1_linear/models.py#L487)

Constructs a ResNet-18 model.

#### Signature

```python
def resnet_deep_qc_18(**kwargs):
    ...
```



## squeezenet_qc

[Show source in models.py:627](../../../../clinicadl/quality_check/t1_linear/models.py#L627)

Constructs a SqueezeNet 1.1 model

#### Arguments

- `pretrained` *bool* - If True, returns a model pre-trained on ImageNet

#### Signature

```python
def squeezenet_qc(pretrained=False, **kwargs):
    ...
```