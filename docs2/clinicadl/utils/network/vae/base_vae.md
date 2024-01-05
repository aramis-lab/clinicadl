# BaseVAE

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Vae](./index.md#vae) /
BaseVAE

> Auto-generated documentation for [clinicadl.utils.network.vae.base_vae](../../../../../clinicadl/utils/network/vae/base_vae.py) module.

- [BaseVAE](#basevae)
  - [BaseVAE](#basevae-1)
    - [BaseVAE().compute_outputs_and_loss](#basevae()compute_outputs_and_loss)
    - [BaseVAE().decode](#basevae()decode)
    - [BaseVAE().encode](#basevae()encode)
    - [BaseVAE().forward](#basevae()forward)
    - [BaseVAE().layers](#basevae()layers)
    - [BaseVAE().predict](#basevae()predict)
    - [BaseVAE().reparameterize](#basevae()reparameterize)

## BaseVAE

[Show source in base_vae.py:6](../../../../../clinicadl/utils/network/vae/base_vae.py#L6)

#### Signature

```python
class BaseVAE(Network):
    def __init__(
        self,
        encoder,
        decoder,
        mu_layer,
        var_layer,
        latent_size,
        gpu=True,
        is_3D=False,
        recons_weight=1,
        KL_weight=1,
    ):
        ...
```

#### See also

- [Network](../network.md#network)

### BaseVAE().compute_outputs_and_loss

[Show source in base_vae.py:47](../../../../../clinicadl/utils/network/vae/base_vae.py#L47)

#### Signature

```python
def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
    ...
```

### BaseVAE().decode

[Show source in base_vae.py:76](../../../../../clinicadl/utils/network/vae/base_vae.py#L76)

#### Signature

```python
def decode(self, z):
    ...
```

### BaseVAE().encode

[Show source in base_vae.py:71](../../../../../clinicadl/utils/network/vae/base_vae.py#L71)

#### Signature

```python
def encode(self, x):
    ...
```

### BaseVAE().forward

[Show source in base_vae.py:42](../../../../../clinicadl/utils/network/vae/base_vae.py#L42)

#### Signature

```python
def forward(self, x):
    ...
```

### BaseVAE().layers

[Show source in base_vae.py:31](../../../../../clinicadl/utils/network/vae/base_vae.py#L31)

#### Signature

```python
@property
def layers(self):
    ...
```

### BaseVAE().predict

[Show source in base_vae.py:38](../../../../../clinicadl/utils/network/vae/base_vae.py#L38)

#### Signature

```python
def predict(self, x):
    ...
```

### BaseVAE().reparameterize

[Show source in base_vae.py:80](../../../../../clinicadl/utils/network/vae/base_vae.py#L80)

#### Signature

```python
def reparameterize(self, mu, logvar):
    ...
```