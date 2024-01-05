# Tsvtools Utils

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Utils](./index.md#utils) /
Tsvtools Utils

> Auto-generated documentation for [clinicadl.utils.tsvtools_utils](../../../clinicadl/utils/tsvtools_utils.py) module.

- [Tsvtools Utils](#tsvtools-utils)
  - [add_demographics](#add_demographics)
  - [after_end_screening](#after_end_screening)
  - [category_conversion](#category_conversion)
  - [chi2](#chi2)
  - [cleaning_nan_diagnoses](#cleaning_nan_diagnoses)
  - [complementary_list](#complementary_list)
  - [df_to_tsv](#df_to_tsv)
  - [extract_baseline](#extract_baseline)
  - [find_label](#find_label)
  - [first_session](#first_session)
  - [last_session](#last_session)
  - [merged_tsv_reader](#merged_tsv_reader)
  - [neighbour_session](#neighbour_session)
  - [next_session](#next_session)
  - [remove_sub_labels](#remove_sub_labels)
  - [remove_unicity](#remove_unicity)
  - [retrieve_longitudinal](#retrieve_longitudinal)

## add_demographics

[Show source in tsvtools_utils.py:121](../../../clinicadl/utils/tsvtools_utils.py#L121)

#### Signature

```python
def add_demographics(df, demographics_df, diagnosis):
    ...
```



## after_end_screening

[Show source in tsvtools_utils.py:43](../../../clinicadl/utils/tsvtools_utils.py#L43)

#### Signature

```python
def after_end_screening(session, session_list):
    ...
```



## category_conversion

[Show source in tsvtools_utils.py:148](../../../clinicadl/utils/tsvtools_utils.py#L148)

#### Signature

```python
def category_conversion(values_list):
    ...
```



## chi2

[Show source in tsvtools_utils.py:106](../../../clinicadl/utils/tsvtools_utils.py#L106)

#### Signature

```python
def chi2(x_test, x_train):
    ...
```



## cleaning_nan_diagnoses

[Show source in tsvtools_utils.py:218](../../../clinicadl/utils/tsvtools_utils.py#L218)

Printing the number of missing diagnoses and filling it partially for ADNI datasets

#### Arguments

- `bids_df` - DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']

#### Returns

cleaned DataFrame

#### Signature

```python
def cleaning_nan_diagnoses(bids_df: pd.DataFrame) -> pd.DataFrame:
    ...
```



## complementary_list

[Show source in tsvtools_utils.py:59](../../../clinicadl/utils/tsvtools_utils.py#L59)

#### Signature

```python
def complementary_list(total_list, sub_list):
    ...
```



## df_to_tsv

[Show source in tsvtools_utils.py:274](../../../clinicadl/utils/tsvtools_utils.py#L274)

Write Dataframe into a TSV file and drop duplicates

Parameters
----------
name: str
    Name of the tsv file
results_path: str (path)
    Path to the folder
df: DataFrame
    DataFrame you want to write in a TSV file.
    Columns must include ["participant_id", "session_id"].
baseline: bool
    If True, ther is only baseline session for each subject.

#### Signature

```python
def df_to_tsv(name: str, results_path: Path, df, baseline: bool = False) -> None:
    ...
```



## extract_baseline

[Show source in tsvtools_utils.py:84](../../../clinicadl/utils/tsvtools_utils.py#L84)

#### Signature

```python
def extract_baseline(diagnosis_df, set_index=True):
    ...
```



## find_label

[Show source in tsvtools_utils.py:157](../../../clinicadl/utils/tsvtools_utils.py#L157)

#### Signature

```python
def find_label(labels_list, target_label):
    ...
```



## first_session

[Show source in tsvtools_utils.py:67](../../../clinicadl/utils/tsvtools_utils.py#L67)

#### Signature

```python
def first_session(subject_df):
    ...
```



## last_session

[Show source in tsvtools_utils.py:53](../../../clinicadl/utils/tsvtools_utils.py#L53)

#### Signature

```python
def last_session(session_list):
    ...
```



## merged_tsv_reader

[Show source in tsvtools_utils.py:15](../../../clinicadl/utils/tsvtools_utils.py#L15)

#### Signature

```python
def merged_tsv_reader(merged_tsv_path: Path):
    ...
```



## neighbour_session

[Show source in tsvtools_utils.py:28](../../../clinicadl/utils/tsvtools_utils.py#L28)

#### Signature

```python
def neighbour_session(session, session_list, neighbour):
    ...
```



## next_session

[Show source in tsvtools_utils.py:73](../../../clinicadl/utils/tsvtools_utils.py#L73)

#### Signature

```python
def next_session(subject_df, session_orig):
    ...
```



## remove_sub_labels

[Show source in tsvtools_utils.py:185](../../../clinicadl/utils/tsvtools_utils.py#L185)

#### Signature

```python
def remove_sub_labels(
    diagnosis_df, sub_labels, diagnosis_df_paths: list[Path], results_path: Path
):
    ...
```



## remove_unicity

[Show source in tsvtools_utils.py:138](../../../clinicadl/utils/tsvtools_utils.py#L138)

Count the values of each class and label all the classes with only one label under the same label.

#### Signature

```python
def remove_unicity(values_list):
    ...
```



## retrieve_longitudinal

[Show source in tsvtools_utils.py:175](../../../clinicadl/utils/tsvtools_utils.py#L175)

#### Signature

```python
def retrieve_longitudinal(df, diagnosis_df):
    ...
```