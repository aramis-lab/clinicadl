# Resnet3d

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Cnn](./index.md#cnn) /
Resnet3d

> Auto-generated documentation for [clinicadl.utils.network.cnn.resnet3D](../../../../../clinicadl/utils/network/cnn/resnet3D.py) module.

- [Resnet3d](#resnet3d)
  - [Flatten](#flatten)
    - [Flatten().forward](#flatten()forward)
  - [ResBlock](#resblock)
    - [ResBlock().forward](#resblock()forward)
  - [ResNetDesigner3D](#resnetdesigner3d)

## Flatten

[Show source in resnet3D.py:5](../../../../../clinicadl/utils/network/cnn/resnet3D.py#L5)

#### Signature

```python
class Flatten(nn.Module):
    ...
```

### Flatten().forward

[Show source in resnet3D.py:6](../../../../../clinicadl/utils/network/cnn/resnet3D.py#L6)

#### Signature

```python
def forward(self, input):
    ...
```



## ResBlock

[Show source in resnet3D.py:10](../../../../../clinicadl/utils/network/cnn/resnet3D.py#L10)

#### Signature

```python
class ResBlock(nn.Module):
    def __init__(self, block_number, input_size):
        ...
```

### ResBlock().forward

[Show source in resnet3D.py:37](../../../../../clinicadl/utils/network/cnn/resnet3D.py#L37)

#### Signature

```python
def forward(self, x):
    ...
```



## ResNetDesigner3D

[Show source in resnet3D.py:49](../../../../../clinicadl/utils/network/cnn/resnet3D.py#L49)

#### Signature

```python
class ResNetDesigner3D(nn.Module):
    def __init__(self, input_size=[1, 169, 208, 179]):
        ...
```