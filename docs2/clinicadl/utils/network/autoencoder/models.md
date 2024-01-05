# Models

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Autoencoder](./index.md#autoencoder) /
Models

> Auto-generated documentation for [clinicadl.utils.network.autoencoder.models](../../../../../clinicadl/utils/network/autoencoder/models.py) module.

- [Models](#models)
  - [AE_Conv4_FC3](#ae_conv4_fc3)
    - [AE_Conv4_FC3.get_dimension](#ae_conv4_fc3get_dimension)
    - [AE_Conv4_FC3.get_input_size](#ae_conv4_fc3get_input_size)
    - [AE_Conv4_FC3.get_task](#ae_conv4_fc3get_task)
  - [AE_Conv5_FC3](#ae_conv5_fc3)
    - [AE_Conv5_FC3.get_dimension](#ae_conv5_fc3get_dimension)
    - [AE_Conv5_FC3.get_input_size](#ae_conv5_fc3get_input_size)
    - [AE_Conv5_FC3.get_task](#ae_conv5_fc3get_task)

## AE_Conv4_FC3

[Show source in models.py:33](../../../../../clinicadl/utils/network/autoencoder/models.py#L33)

Autoencoder derived from the convolutional part of CNN Conv4_FC3.

#### Signature

```python
class AE_Conv4_FC3(AutoEncoder):
    def __init__(self, input_size, gpu=True):
        ...
```

#### See also

- [AutoEncoder](../sub_network.md#autoencoder)

### AE_Conv4_FC3.get_dimension

[Show source in models.py:51](../../../../../clinicadl/utils/network/autoencoder/models.py#L51)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### AE_Conv4_FC3.get_input_size

[Show source in models.py:47](../../../../../clinicadl/utils/network/autoencoder/models.py#L47)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### AE_Conv4_FC3.get_task

[Show source in models.py:55](../../../../../clinicadl/utils/network/autoencoder/models.py#L55)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## AE_Conv5_FC3

[Show source in models.py:6](../../../../../clinicadl/utils/network/autoencoder/models.py#L6)

Autoencoder derived from the convolutional part of CNN Conv5_FC3.

#### Signature

```python
class AE_Conv5_FC3(AutoEncoder):
    def __init__(self, input_size, gpu=True):
        ...
```

#### See also

- [AutoEncoder](../sub_network.md#autoencoder)

### AE_Conv5_FC3.get_dimension

[Show source in models.py:24](../../../../../clinicadl/utils/network/autoencoder/models.py#L24)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### AE_Conv5_FC3.get_input_size

[Show source in models.py:20](../../../../../clinicadl/utils/network/autoencoder/models.py#L20)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### AE_Conv5_FC3.get_task

[Show source in models.py:28](../../../../../clinicadl/utils/network/autoencoder/models.py#L28)

#### Signature

```python
@staticmethod
def get_task():
    ...
```