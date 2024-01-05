# Sub Network

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Network](./index.md#network) /
Sub Network

> Auto-generated documentation for [clinicadl.utils.network.sub_network](../../../../clinicadl/utils/network/sub_network.py) module.

- [Sub Network](#sub-network)
  - [AutoEncoder](#autoencoder)
    - [AutoEncoder().compute_outputs_and_loss](#autoencoder()compute_outputs_and_loss)
    - [AutoEncoder().forward](#autoencoder()forward)
    - [AutoEncoder().layers](#autoencoder()layers)
    - [AutoEncoder().predict](#autoencoder()predict)
    - [AutoEncoder().transfer_weights](#autoencoder()transfer_weights)
  - [CNN](#cnn)
    - [CNN().compute_outputs_and_loss](#cnn()compute_outputs_and_loss)
    - [CNN().forward](#cnn()forward)
    - [CNN().layers](#cnn()layers)
    - [CNN().predict](#cnn()predict)
    - [CNN().transfer_weights](#cnn()transfer_weights)

## AutoEncoder

[Show source in sub_network.py:19](../../../../clinicadl/utils/network/sub_network.py#L19)

#### Signature

```python
class AutoEncoder(Network):
    def __init__(self, encoder, decoder, gpu=False):
        ...
```

#### See also

- [Network](./network.md#network)

### AutoEncoder().compute_outputs_and_loss

[Show source in sub_network.py:82](../../../../clinicadl/utils/network/sub_network.py#L82)

#### Signature

```python
def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
    ...
```

### AutoEncoder().forward

[Show source in sub_network.py:50](../../../../clinicadl/utils/network/sub_network.py#L50)

#### Signature

```python
def forward(self, x):
    ...
```

### AutoEncoder().layers

[Show source in sub_network.py:25](../../../../clinicadl/utils/network/sub_network.py#L25)

#### Signature

```python
@property
def layers(self):
    ...
```

### AutoEncoder().predict

[Show source in sub_network.py:46](../../../../clinicadl/utils/network/sub_network.py#L46)

#### Signature

```python
def predict(self, x):
    ...
```

### AutoEncoder().transfer_weights

[Show source in sub_network.py:29](../../../../clinicadl/utils/network/sub_network.py#L29)

#### Signature

```python
def transfer_weights(self, state_dict, transfer_class):
    ...
```



## CNN

[Show source in sub_network.py:91](../../../../clinicadl/utils/network/sub_network.py#L91)

#### Signature

```python
class CNN(Network):
    def __init__(self, convolutions, fc, n_classes, gpu=False):
        ...
```

#### See also

- [Network](./network.md#network)

### CNN().compute_outputs_and_loss

[Show source in sub_network.py:126](../../../../clinicadl/utils/network/sub_network.py#L126)

#### Signature

```python
def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
    ...
```

### CNN().forward

[Show source in sub_network.py:119](../../../../clinicadl/utils/network/sub_network.py#L119)

#### Signature

```python
def forward(self, x):
    ...
```

### CNN().layers

[Show source in sub_network.py:98](../../../../clinicadl/utils/network/sub_network.py#L98)

#### Signature

```python
@property
def layers(self):
    ...
```

### CNN().predict

[Show source in sub_network.py:123](../../../../clinicadl/utils/network/sub_network.py#L123)

#### Signature

```python
def predict(self, x):
    ...
```

### CNN().transfer_weights

[Show source in sub_network.py:102](../../../../clinicadl/utils/network/sub_network.py#L102)

#### Signature

```python
def transfer_weights(self, state_dict, transfer_class):
    ...
```