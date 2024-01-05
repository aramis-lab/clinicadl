# Network Utils

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Network](./index.md#network) /
Network Utils

> Auto-generated documentation for [clinicadl.utils.network.network_utils](../../../../clinicadl/utils/network/network_utils.py) module.

- [Network Utils](#network-utils)
  - [CropMaxUnpool2d](#cropmaxunpool2d)
    - [CropMaxUnpool2d().forward](#cropmaxunpool2d()forward)
  - [CropMaxUnpool3d](#cropmaxunpool3d)
    - [CropMaxUnpool3d().forward](#cropmaxunpool3d()forward)
  - [PadMaxPool2d](#padmaxpool2d)
    - [PadMaxPool2d().forward](#padmaxpool2d()forward)
    - [PadMaxPool2d().set_new_return](#padmaxpool2d()set_new_return)
  - [PadMaxPool3d](#padmaxpool3d)
    - [PadMaxPool3d().forward](#padmaxpool3d()forward)
    - [PadMaxPool3d().set_new_return](#padmaxpool3d()set_new_return)
  - [Reshape](#reshape)
    - [Reshape().forward](#reshape()forward)

## CropMaxUnpool2d

[Show source in network_utils.py:113](../../../../clinicadl/utils/network/network_utils.py#L113)

#### Signature

```python
class CropMaxUnpool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        ...
```

### CropMaxUnpool2d().forward

[Show source in network_utils.py:118](../../../../clinicadl/utils/network/network_utils.py#L118)

#### Signature

```python
def forward(self, f_maps, indices, padding=None):
    ...
```



## CropMaxUnpool3d

[Show source in network_utils.py:97](../../../../clinicadl/utils/network/network_utils.py#L97)

#### Signature

```python
class CropMaxUnpool3d(nn.Module):
    def __init__(self, kernel_size, stride):
        ...
```

### CropMaxUnpool3d().forward

[Show source in network_utils.py:102](../../../../clinicadl/utils/network/network_utils.py#L102)

#### Signature

```python
def forward(self, f_maps, indices, padding=None):
    ...
```



## PadMaxPool2d

[Show source in network_utils.py:57](../../../../clinicadl/utils/network/network_utils.py#L57)

#### Signature

```python
class PadMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        ...
```

### PadMaxPool2d().forward

[Show source in network_utils.py:72](../../../../clinicadl/utils/network/network_utils.py#L72)

#### Signature

```python
def forward(self, f_maps):
    ...
```

### PadMaxPool2d().set_new_return

[Show source in network_utils.py:67](../../../../clinicadl/utils/network/network_utils.py#L67)

#### Signature

```python
def set_new_return(self, return_indices=True, return_pad=True):
    ...
```



## PadMaxPool3d

[Show source in network_utils.py:17](../../../../clinicadl/utils/network/network_utils.py#L17)

#### Signature

```python
class PadMaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        ...
```

### PadMaxPool3d().forward

[Show source in network_utils.py:32](../../../../clinicadl/utils/network/network_utils.py#L32)

#### Signature

```python
def forward(self, f_maps):
    ...
```

### PadMaxPool3d().set_new_return

[Show source in network_utils.py:27](../../../../clinicadl/utils/network/network_utils.py#L27)

#### Signature

```python
def set_new_return(self, return_indices=True, return_pad=True):
    ...
```



## Reshape

[Show source in network_utils.py:8](../../../../clinicadl/utils/network/network_utils.py#L8)

#### Signature

```python
class Reshape(nn.Module):
    def __init__(self, size):
        ...
```

### Reshape().forward

[Show source in network_utils.py:13](../../../../clinicadl/utils/network/network_utils.py#L13)

#### Signature

```python
def forward(self, input):
    ...
```