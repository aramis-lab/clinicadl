# Cnn Transformer

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Autoencoder](./index.md#autoencoder) /
Cnn Transformer

> Auto-generated documentation for [clinicadl.utils.network.autoencoder.cnn_transformer](../../../../../clinicadl/utils/network/autoencoder/cnn_transformer.py) module.

- [Cnn Transformer](#cnn-transformer)
  - [CNN_Transformer](#cnn_transformer)
    - [CNN_Transformer().construct_inv_layers](#cnn_transformer()construct_inv_layers)
    - [CNN_Transformer.replace_relu](#cnn_transformerreplace_relu)

## CNN_Transformer

[Show source in cnn_transformer.py:14](../../../../../clinicadl/utils/network/autoencoder/cnn_transformer.py#L14)

#### Signature

```python
class CNN_Transformer(nn.Module):
    def __init__(self, model=None):
        ...
```

### CNN_Transformer().construct_inv_layers

[Show source in cnn_transformer.py:43](../../../../../clinicadl/utils/network/autoencoder/cnn_transformer.py#L43)

Implements the decoder part from the CNN. The decoder part is the symmetrical list of the encoder
in which some layers are replaced by their transpose counterpart.
ConvTranspose and ReLU layers are inverted in the end.

#### Arguments

- `model` - (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.

#### Returns

(Module) decoder part of the Autoencoder

#### Signature

```python
def construct_inv_layers(self, model):
    ...
```

### CNN_Transformer.replace_relu

[Show source in cnn_transformer.py:96](../../../../../clinicadl/utils/network/autoencoder/cnn_transformer.py#L96)

Invert convolutional and ReLU layers (give empirical better results)

#### Arguments

- `inv_layers` - (list) list of the layers of decoder part of the Auto-Encoder

#### Returns

(list) the layers with the inversion

#### Signature

```python
@staticmethod
def replace_relu(inv_layers):
    ...
```