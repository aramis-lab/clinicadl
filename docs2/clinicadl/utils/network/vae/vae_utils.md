# Vae Utils

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Vae](./index.md#vae) /
Vae Utils

> Auto-generated documentation for [clinicadl.utils.network.vae.vae_utils](../../../../../clinicadl/utils/network/vae/vae_utils.py) module.

- [Vae Utils](#vae-utils)
  - [DecoderLayer2D](#decoderlayer2d)
    - [DecoderLayer2D().forward](#decoderlayer2d()forward)
  - [DecoderLayer3D](#decoderlayer3d)
    - [DecoderLayer3D().forward](#decoderlayer3d()forward)
  - [EncoderLayer2D](#encoderlayer2d)
    - [EncoderLayer2D().forward](#encoderlayer2d()forward)
  - [EncoderLayer3D](#encoderlayer3d)
    - [EncoderLayer3D().forward](#encoderlayer3d()forward)
  - [Flatten](#flatten)
    - [Flatten().forward](#flatten()forward)
  - [Unflatten2D](#unflatten2d)
    - [Unflatten2D().forward](#unflatten2d()forward)
  - [Unflatten3D](#unflatten3d)
    - [Unflatten3D().forward](#unflatten3d()forward)
  - [VAE_Decoder](#vae_decoder)
    - [VAE_Decoder().forward](#vae_decoder()forward)
  - [VAE_Encoder](#vae_encoder)
    - [VAE_Encoder().forward](#vae_encoder()forward)

## DecoderLayer2D

[Show source in vae_utils.py:40](../../../../../clinicadl/utils/network/vae/vae_utils.py#L40)

Class defining the decoder's part of the Autoencoder.
This layer is composed of one 2D transposed convolutional layer,
a batch normalization layer with a relu activation function.

#### Signature

```python
class DecoderLayer2D(nn.Module):
    def __init__(
        self, input_channels, output_channels, kernel_size=4, stride=2, padding=1
    ):
        ...
```

### DecoderLayer2D().forward

[Show source in vae_utils.py:63](../../../../../clinicadl/utils/network/vae/vae_utils.py#L63)

#### Signature

```python
def forward(self, x):
    ...
```



## DecoderLayer3D

[Show source in vae_utils.py:97](../../../../../clinicadl/utils/network/vae/vae_utils.py#L97)

Class defining the decoder's part of the Autoencoder.
This layer is composed of one 3D transposed convolutional layer,
a batch normalization layer with a relu activation function.

#### Signature

```python
class DecoderLayer3D(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
    ):
        ...
```

### DecoderLayer3D().forward

[Show source in vae_utils.py:127](../../../../../clinicadl/utils/network/vae/vae_utils.py#L127)

#### Signature

```python
def forward(self, x):
    ...
```



## EncoderLayer2D

[Show source in vae_utils.py:6](../../../../../clinicadl/utils/network/vae/vae_utils.py#L6)

Class defining the encoder's part of the Autoencoder.
This layer is composed of one 2D convolutional layer,
a batch normalization layer with a leaky relu
activation function.

#### Signature

```python
class EncoderLayer2D(nn.Module):
    def __init__(
        self, input_channels, output_channels, kernel_size=4, stride=2, padding=1
    ):
        ...
```

### EncoderLayer2D().forward

[Show source in vae_utils.py:35](../../../../../clinicadl/utils/network/vae/vae_utils.py#L35)

#### Signature

```python
def forward(self, x):
    ...
```



## EncoderLayer3D

[Show source in vae_utils.py:68](../../../../../clinicadl/utils/network/vae/vae_utils.py#L68)

Class defining the encoder's part of the Autoencoder.
This layer is composed of one 3D convolutional layer,
a batch normalization layer with a leaky relu
activation function.

#### Signature

```python
class EncoderLayer3D(nn.Module):
    def __init__(
        self, input_channels, output_channels, kernel_size=4, stride=2, padding=1
    ):
        ...
```

### EncoderLayer3D().forward

[Show source in vae_utils.py:92](../../../../../clinicadl/utils/network/vae/vae_utils.py#L92)

#### Signature

```python
def forward(self, x):
    ...
```



## Flatten

[Show source in vae_utils.py:132](../../../../../clinicadl/utils/network/vae/vae_utils.py#L132)

#### Signature

```python
class Flatten(nn.Module):
    ...
```

### Flatten().forward

[Show source in vae_utils.py:133](../../../../../clinicadl/utils/network/vae/vae_utils.py#L133)

#### Signature

```python
def forward(self, input):
    ...
```



## Unflatten2D

[Show source in vae_utils.py:137](../../../../../clinicadl/utils/network/vae/vae_utils.py#L137)

#### Signature

```python
class Unflatten2D(nn.Module):
    def __init__(self, channel, height, width):
        ...
```

### Unflatten2D().forward

[Show source in vae_utils.py:144](../../../../../clinicadl/utils/network/vae/vae_utils.py#L144)

#### Signature

```python
def forward(self, input):
    ...
```



## Unflatten3D

[Show source in vae_utils.py:148](../../../../../clinicadl/utils/network/vae/vae_utils.py#L148)

#### Signature

```python
class Unflatten3D(nn.Module):
    def __init__(self, channel, height, width, depth):
        ...
```

### Unflatten3D().forward

[Show source in vae_utils.py:156](../../../../../clinicadl/utils/network/vae/vae_utils.py#L156)

#### Signature

```python
def forward(self, input):
    ...
```



## VAE_Decoder

[Show source in vae_utils.py:251](../../../../../clinicadl/utils/network/vae/vae_utils.py#L251)

#### Signature

```python
class VAE_Decoder(nn.Module):
    def __init__(
        self,
        input_shape,
        latent_size,
        n_conv=4,
        last_layer_channels=32,
        latent_dim=1,
        feature_size=1024,
        padding=None,
    ):
        ...
```

### VAE_Decoder().forward

[Show source in vae_utils.py:350](../../../../../clinicadl/utils/network/vae/vae_utils.py#L350)

#### Signature

```python
def forward(self, z):
    ...
```



## VAE_Encoder

[Show source in vae_utils.py:162](../../../../../clinicadl/utils/network/vae/vae_utils.py#L162)

#### Signature

```python
class VAE_Encoder(nn.Module):
    def __init__(
        self,
        input_shape,
        n_conv=4,
        first_layer_channels=32,
        latent_dim=1,
        feature_size=1024,
    ):
        ...
```

### VAE_Encoder().forward

[Show source in vae_utils.py:246](../../../../../clinicadl/utils/network/vae/vae_utils.py#L246)

#### Signature

```python
def forward(self, x):
    ...
```