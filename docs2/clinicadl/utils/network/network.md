# Network

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Network](./index.md#network) /
Network

> Auto-generated documentation for [clinicadl.utils.network.network](../../../../clinicadl/utils/network/network.py) module.

- [Network](#network)
  - [Network](#network-1)
    - [Network().compute_outputs_and_loss](#network()compute_outputs_and_loss)
    - [Network().forward](#network()forward)
    - [Network.get_dimension](#networkget_dimension)
    - [Network.get_input_size](#networkget_input_size)
    - [Network.get_task](#networkget_task)
    - [Network().layers](#network()layers)
    - [Network().predict](#network()predict)
    - [Network().transfer_weights](#network()transfer_weights)

## Network

[Show source in network.py:9](../../../../clinicadl/utils/network/network.py#L9)

Abstract Template for all networks used in ClinicaDL

#### Signature

```python
class Network(nn.Module):
    def __init__(self, gpu=True):
        ...
```

### Network().compute_outputs_and_loss

[Show source in network.py:93](../../../../clinicadl/utils/network/network.py#L93)

#### Signature

```python
@abc.abstractmethod
def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
    ...
```

### Network().forward

[Show source in network.py:89](../../../../clinicadl/utils/network/network.py#L89)

#### Signature

```python
@abc.abstractmethod
def forward(self, x):
    ...
```

### Network.get_dimension

[Show source in network.py:65](../../../../clinicadl/utils/network/network.py#L65)

This static method is used for list_models command.
Return '2D', '3D' or '2D and 3D'

#### Signature

```python
@staticmethod
@abc.abstractmethod
def get_dimension() -> str:
    ...
```

### Network.get_input_size

[Show source in network.py:56](../../../../clinicadl/utils/network/network.py#L56)

This static method is used for list_models command.
Must return the shape of the input size expected (C@HxW or C@HxWxD) for each architecture.

#### Signature

```python
@staticmethod
@abc.abstractmethod
def get_input_size() -> str:
    ...
```

### Network.get_task

[Show source in network.py:73](../../../../clinicadl/utils/network/network.py#L73)

This static method is used for list_models command.
Return the list of tasks for which the model is made.

#### Signature

```python
@staticmethod
@abc.abstractmethod
def get_task() -> list:
    ...
```

### Network().layers

[Show source in network.py:81](../../../../clinicadl/utils/network/network.py#L81)

#### Signature

```python
@abc.abstractproperty
def layers(self):
    ...
```

### Network().predict

[Show source in network.py:85](../../../../clinicadl/utils/network/network.py#L85)

#### Signature

```python
@abc.abstractmethod
def predict(self, x):
    ...
```

### Network().transfer_weights

[Show source in network.py:97](../../../../clinicadl/utils/network/network.py#L97)

#### Signature

```python
def transfer_weights(self, state_dict, transfer_class):
    ...
```