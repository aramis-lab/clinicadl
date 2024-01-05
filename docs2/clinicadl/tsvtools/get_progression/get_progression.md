# Get Progression

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Get Progression](./index.md#get-progression) /
Get Progression

> Auto-generated documentation for [clinicadl.tsvtools.get_progression.get_progression](../../../../clinicadl/tsvtools/get_progression/get_progression.py) module.

- [Get Progression](#get-progression)
  - [get_progression](#get_progression)

## get_progression

[Show source in get_progression.py:18](../../../../clinicadl/tsvtools/get_progression/get_progression.py#L18)

A method to get the progression for each sessions depending on their stability on the time horizon
Outputs are written in data_tsv

Parameters
----------
data_tsv: str (path)
    Path to a tsv file with columns including ["participants_id", "session_id", "dignosis"]
horizon_time: int
    Time horizon in months
stability_dict: dict
    Dictionnary explaining the progression of the disease. If None, it uses the Alzheimer's one : {CN: 0, MCI: 1, AD: 2}

#### Signature

```python
def get_progression(data_tsv: Path, horizon_time: int = 36, stability_dict: dict = None):
    ...
```