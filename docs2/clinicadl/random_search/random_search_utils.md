# Random Search Utils

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Random Search](./index.md#random-search) /
Random Search Utils

> Auto-generated documentation for [clinicadl.random_search.random_search_utils](../../../clinicadl/random_search/random_search_utils.py) module.

- [Random Search Utils](#random-search-utils)
  - [get_space_dict](#get_space_dict)
  - [random_conv_sampling](#random_conv_sampling)
  - [random_sampling](#random_sampling)
  - [sampling_fn](#sampling_fn)
  - [update_channels](#update_channels)

## get_space_dict

[Show source in random_search_utils.py:13](../../../clinicadl/random_search/random_search_utils.py#L13)

Transforms the TOML dictionary in one dimension dictionary.

#### Signature

```python
def get_space_dict(launch_directory: Path) -> Dict[str, Any]:
    ...
```



## random_conv_sampling

[Show source in random_search_utils.py:165](../../../clinicadl/random_search/random_search_utils.py#L165)

Generate random parameters for a random architecture (convolutional part).

#### Arguments

    - `rs_options` - parameters of the random search
Returns
    parameters of the convolutions

#### Signature

```python
def random_conv_sampling(rs_options: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ...
```



## random_sampling

[Show source in random_search_utils.py:101](../../../clinicadl/random_search/random_search_utils.py#L101)

Samples all the hyperparameters of the model.

#### Arguments

- `rs_options` - parameters of the random search

#### Returns

options updated to train the model generated randomly

#### Signature

```python
def random_sampling(rs_options: Dict[str, Any]) -> Dict[str, Any]:
    ...
```



## sampling_fn

[Show source in random_search_utils.py:77](../../../clinicadl/random_search/random_search_utils.py#L77)

#### Signature

```python
def sampling_fn(value, sampling_type: str):
    ...
```



## update_channels

[Show source in random_search_utils.py:196](../../../clinicadl/random_search/random_search_utils.py#L196)

#### Signature

```python
def update_channels(out_channels: int, channels_limit: int = 512) -> Tuple[int, int]:
    ...
```