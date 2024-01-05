# Vanilla Vae

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Vae](./index.md#vae) /
Vanilla Vae

> Auto-generated documentation for [clinicadl.utils.network.vae.vanilla_vae](../../../../../clinicadl/utils/network/vae/vanilla_vae.py) module.

- [Vanilla Vae](#vanilla-vae)
  - [Vanilla3DVAE](#vanilla3dvae)
    - [Vanilla3DVAE.get_dimension](#vanilla3dvaeget_dimension)
    - [Vanilla3DVAE.get_input_size](#vanilla3dvaeget_input_size)
    - [Vanilla3DVAE.get_task](#vanilla3dvaeget_task)
  - [Vanilla3DdenseVAE](#vanilla3ddensevae)
    - [Vanilla3DdenseVAE.get_dimension](#vanilla3ddensevaeget_dimension)
    - [Vanilla3DdenseVAE.get_input_size](#vanilla3ddensevaeget_input_size)
    - [Vanilla3DdenseVAE.get_task](#vanilla3ddensevaeget_task)
  - [VanillaDenseVAE](#vanilladensevae)
    - [VanillaDenseVAE.get_dimension](#vanilladensevaeget_dimension)
    - [VanillaDenseVAE.get_input_size](#vanilladensevaeget_input_size)
    - [VanillaDenseVAE.get_task](#vanilladensevaeget_task)
  - [VanillaSpatialVAE](#vanillaspatialvae)
    - [VanillaSpatialVAE.get_dimension](#vanillaspatialvaeget_dimension)
    - [VanillaSpatialVAE.get_input_size](#vanillaspatialvaeget_input_size)
    - [VanillaSpatialVAE.get_task](#vanillaspatialvaeget_task)

## Vanilla3DVAE

[Show source in vanilla_vae.py:149](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L149)

This network is a 3D convolutional variational autoencoder with a spacial latent space.

reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
https://arxiv.org/abs/1312.6114

#### Signature

```python
class Vanilla3DVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        latent_space_size=256,
        feature_size=1024,
        recons_weight=1,
        KL_weight=1,
        gpu=True,
    ):
        ...
```

#### See also

- [BaseVAE](./base_vae.md#basevae)

### Vanilla3DVAE.get_dimension

[Show source in vanilla_vae.py:280](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L280)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### Vanilla3DVAE.get_input_size

[Show source in vanilla_vae.py:276](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L276)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### Vanilla3DVAE.get_task

[Show source in vanilla_vae.py:284](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L284)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## Vanilla3DdenseVAE

[Show source in vanilla_vae.py:289](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L289)

This network is a 3D convolutional variational autoencoder with a dense latent space.

reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
https://arxiv.org/abs/1312.6114

#### Signature

```python
class Vanilla3DdenseVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        latent_space_size=256,
        feature_size=1024,
        n_conv=4,
        io_layer_channels=8,
        recons_weight=1,
        KL_weight=1,
        gpu=True,
    ):
        ...
```

#### See also

- [BaseVAE](./base_vae.md#basevae)

### Vanilla3DdenseVAE.get_dimension

[Show source in vanilla_vae.py:432](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L432)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### Vanilla3DdenseVAE.get_input_size

[Show source in vanilla_vae.py:428](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L428)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### Vanilla3DdenseVAE.get_task

[Show source in vanilla_vae.py:436](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L436)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## VanillaDenseVAE

[Show source in vanilla_vae.py:15](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L15)

This network is a 2D convolutional variational autoencoder with a dense latent space.

reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
https://arxiv.org/abs/1312.6114

#### Signature

```python
class VanillaDenseVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        latent_space_size=128,
        feature_size=1024,
        recons_weight=1,
        KL_weight=1,
        gpu=True,
    ):
        ...
```

#### See also

- [BaseVAE](./base_vae.md#basevae)

### VanillaDenseVAE.get_dimension

[Show source in vanilla_vae.py:70](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L70)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### VanillaDenseVAE.get_input_size

[Show source in vanilla_vae.py:66](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L66)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### VanillaDenseVAE.get_task

[Show source in vanilla_vae.py:74](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L74)

#### Signature

```python
@staticmethod
def get_task():
    ...
```



## VanillaSpatialVAE

[Show source in vanilla_vae.py:79](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L79)

This network is a 3D convolutional variational autoencoder with a spacial latent space.

reference: Diederik P Kingma et al., Auto-Encoding Variational Bayes.
https://arxiv.org/abs/1312.6114

#### Signature

```python
class VanillaSpatialVAE(BaseVAE):
    def __init__(
        self,
        input_size,
        latent_space_size=128,
        feature_size=1024,
        recons_weight=1,
        KL_weight=1,
        gpu=True,
    ):
        ...
```

#### See also

- [BaseVAE](./base_vae.md#basevae)

### VanillaSpatialVAE.get_dimension

[Show source in vanilla_vae.py:140](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L140)

#### Signature

```python
@staticmethod
def get_dimension():
    ...
```

### VanillaSpatialVAE.get_input_size

[Show source in vanilla_vae.py:136](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L136)

#### Signature

```python
@staticmethod
def get_input_size():
    ...
```

### VanillaSpatialVAE.get_task

[Show source in vanilla_vae.py:144](../../../../../clinicadl/utils/network/vae/vanilla_vae.py#L144)

#### Signature

```python
@staticmethod
def get_task():
    ...
```