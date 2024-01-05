# Seed

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Utils](./index.md#utils) /
Seed

> Auto-generated documentation for [clinicadl.utils.seed](../../../clinicadl/utils/seed.py) module.

- [Seed](#seed)
  - [get_seed](#get_seed)
  - [pl_worker_init_function](#pl_worker_init_function)
  - [seed_everything](#seed_everything)

## get_seed

[Show source in seed.py:47](../../../clinicadl/utils/seed.py#L47)

#### Signature

```python
def get_seed(seed: int = None) -> int:
    ...
```



## pl_worker_init_function

[Show source in seed.py:21](../../../clinicadl/utils/seed.py#L21)

The worker_init_fn that Lightning automatically adds to your dataloader if you previously set
set the seed with ``seed_everything(seed, workers=True)``.
See also the PyTorch documentation on
`randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.

#### Signature

```python
def pl_worker_init_function(worker_id: int) -> None:
    ...
```



## seed_everything

[Show source in seed.py:57](../../../clinicadl/utils/seed.py#L57)

Function that sets seed for pseudo-random number generators in:
pytorch, numpy, python.random

Adapted from pytorch-lightning
https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything

#### Arguments

- `seed` *int* - Value of the seed for all pseudo-random number generators
- `deterministic` *bool* - If set to True will raise an error if non-deterministic behaviour is encountered
- `compensation` *str* - Chooses which computational aspect is affected when deterministic is set to True.
    Must be chosen between time and memory.

#### Raises

- `ClinicaDLConfigurationError` - if compensation is not in {"time", "memory"}.
- `RuntimeError` - if a non-deterministic behaviour was encountered.

#### Signature

```python
def seed_everything(seed, deterministic=False, compensation="memory") -> None:
    ...
```