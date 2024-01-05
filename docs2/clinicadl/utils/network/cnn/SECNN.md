# Secnn

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Cnn](./index.md#cnn) /
Secnn

> Auto-generated documentation for [clinicadl.utils.network.cnn.SECNN](../../../../../clinicadl/utils/network/cnn/SECNN.py) module.

- [Secnn](#secnn)
  - [Flatten](#flatten)
    - [Flatten().forward](#flatten()forward)
  - [ResBlock_SE](#resblock_se)
    - [ResBlock_SE().forward](#resblock_se()forward)
  - [SECNNDesigner3D](#secnndesigner3d)
  - [SE_Blocks](#se_blocks)
    - [SE_Blocks().forward](#se_blocks()forward)

## Flatten

[Show source in SECNN.py:5](../../../../../clinicadl/utils/network/cnn/SECNN.py#L5)

#### Signature

```python
class Flatten(nn.Module):
    ...
```

### Flatten().forward

[Show source in SECNN.py:6](../../../../../clinicadl/utils/network/cnn/SECNN.py#L6)

#### Signature

```python
def forward(self, input):
    ...
```



## ResBlock_SE

[Show source in SECNN.py:47](../../../../../clinicadl/utils/network/cnn/SECNN.py#L47)

#### Signature

```python
class ResBlock_SE(nn.Module):
    def __init__(self, block_number, input_size, num_channels, ration_channel=8):
        ...
```

### ResBlock_SE().forward

[Show source in SECNN.py:76](../../../../../clinicadl/utils/network/cnn/SECNN.py#L76)

#### Signature

```python
def forward(self, x):
    ...
```



## SECNNDesigner3D

[Show source in SECNN.py:89](../../../../../clinicadl/utils/network/cnn/SECNN.py#L89)

#### Signature

```python
class SECNNDesigner3D(nn.Module):
    def __init__(self, input_size=[1, 169, 208, 179]):
        ...
```



## SE_Blocks

[Show source in SECNN.py:10](../../../../../clinicadl/utils/network/cnn/SECNN.py#L10)

#### Signature

```python
class SE_Blocks(nn.Module):
    def __init__(self, num_channels, ratio_channel):
        ...
```

### SE_Blocks().forward

[Show source in SECNN.py:21](../../../../../clinicadl/utils/network/cnn/SECNN.py#L21)

Parameters
----------
input_tensor: pt tensor
    X, shape = (batch_size, num_channels, D, H, W)

Returns
-------
output_tensor: pt tensor

#### Signature

```python
def forward(self, input_tensor):
    ...
```