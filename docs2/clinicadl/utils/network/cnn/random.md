# Random

[Clinicadl Index](../../../../README.md#clinicadl-index) /
[Clinicadl](../../../index.md#clinicadl) /
[Utils](../../index.md#utils) /
[Network](../index.md#network) /
[Cnn](./index.md#cnn) /
Random

> Auto-generated documentation for [clinicadl.utils.network.cnn.random](../../../../../clinicadl/utils/network/cnn/random.py) module.

- [Random](#random)
  - [RandomArchitecture](#randomarchitecture)
    - [RandomArchitecture().append_normalization_layer](#randomarchitecture()append_normalization_layer)
    - [RandomArchitecture().define_convolutional_block](#randomarchitecture()define_convolutional_block)
    - [RandomArchitecture.define_fc_layer](#randomarchitecturedefine_fc_layer)
    - [RandomArchitecture.fc_dict_design](#randomarchitecturefc_dict_design)
    - [RandomArchitecture.recursive_init](#randomarchitecturerecursive_init)
    - [RandomArchitecture().return_layers_dict](#randomarchitecture()return_layers_dict)

## RandomArchitecture

[Show source in random.py:8](../../../../../clinicadl/utils/network/cnn/random.py#L8)

#### Signature

```python
class RandomArchitecture(CNN):
    def __init__(
        self,
        convolutions_dict,
        n_fcblocks,
        input_size,
        dropout=0.5,
        network_normalization="BatchNorm",
        output_size=2,
        gpu=True,
    ):
        ...
```

#### See also

- [CNN](../sub_network.md#cnn)

### RandomArchitecture().append_normalization_layer

[Show source in random.py:115](../../../../../clinicadl/utils/network/cnn/random.py#L115)

Appends or not a normalization layer to a convolutional block depending on network attributes.

#### Arguments

- `conv_block` - (list) list of the modules of the convolutional block
- `num_features` - (int) number of features to normalize

#### Returns

(list) the updated convolutional block

#### Signature

```python
def append_normalization_layer(self, conv_block, num_features):
    ...
```

### RandomArchitecture().define_convolutional_block

[Show source in random.py:61](../../../../../clinicadl/utils/network/cnn/random.py#L61)

Design a convolutional block from the dictionary conv_dict.

#### Arguments

- `conv_dict` - (dict) A dictionary with the specifications to build a convolutional block
- n_conv (int) number of convolutional layers in the block
- in_channels (int) number of input channels
- out_channels (int) number of output channels (2 * in_channels or threshold = 512)
- d_reduction (String) "MaxPooling" or "stride"

#### Returns

(nn.Module) a list of modules in a nn.Sequential list

#### Signature

```python
def define_convolutional_block(self, conv_dict):
    ...
```

### RandomArchitecture.define_fc_layer

[Show source in random.py:157](../../../../../clinicadl/utils/network/cnn/random.py#L157)

Implement the FC block from the dictionary fc_dict.

#### Arguments

- `fc_dict` - (dict) A dictionary with the specifications to build a FC block
- in_features (int) number of input neurones
- out_features (int) number of output neurones
- `last_block` - (bool) indicates if the current FC layer is the last one of the network.

#### Returns

(nn.Module) a list of modules in a nn.Sequential list

#### Signature

```python
@staticmethod
def define_fc_layer(fc_dict, last_block=False):
    ...
```

### RandomArchitecture.fc_dict_design

[Show source in random.py:191](../../../../../clinicadl/utils/network/cnn/random.py#L191)

Sample parameters for a random architecture (FC part).

#### Arguments

- `n_fcblocks` - (int) number of fully connected blocks in the architecture.
- `convolutions` - (dict) parameters of the convolutional part.
- `initial_shape` - (array_like) shape of the initial input.
- `n_classes` - (int) number of classes in the classification problem.

#### Returns

(dict) parameters of the architecture
(list) the shape of the flattened layer

#### Signature

```python
@staticmethod
def fc_dict_design(n_fcblocks, convolutions, initial_shape, n_classes=2):
    ...
```

### RandomArchitecture.recursive_init

[Show source in random.py:180](../../../../../clinicadl/utils/network/cnn/random.py#L180)

#### Signature

```python
@staticmethod
def recursive_init(layer):
    ...
```

### RandomArchitecture().return_layers_dict

[Show source in random.py:136](../../../../../clinicadl/utils/network/cnn/random.py#L136)

#### Signature

```python
def return_layers_dict(self):
    ...
```