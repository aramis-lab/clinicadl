# Get Labels

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Get Labels](./index.md#get-labels) /
Get Labels

> Auto-generated documentation for [clinicadl.tsvtools.get_labels.get_labels](../../../../clinicadl/tsvtools/get_labels/get_labels.py) module.

- [Get Labels](#get-labels)
  - [apply_restriction](#apply_restriction)
  - [diagnosis_removal](#diagnosis_removal)
  - [get_labels](#get_labels)
  - [infer_or_drop_diagnosis](#infer_or_drop_diagnosis)
  - [mod_selection](#mod_selection)
  - [remove_unique_session](#remove_unique_session)

## apply_restriction

[Show source in get_labels.py:208](../../../../clinicadl/tsvtools/get_labels/get_labels.py#L208)

Application of a restriction (for example after the removal of some subjects after a preprocessing pipeline)

Parameters
----------
bids_df: DataFrame
    Columns must include ['participant_id', 'session_id', 'diagnosis']
restriction_path: str (path)
    Path to a tsv file with columns including ['participant_id', 'session_id', 'diagnosis'] including
    all the sessions that can be included

Returns
-------
bids_copy_df: DataFrame
    Cleaned copy of the input bids_df

#### Signature

```python
def apply_restriction(bids_df: pd.DataFrame, restriction_path: Path) -> pd.DataFrame:
    ...
```



## diagnosis_removal

[Show source in get_labels.py:177](../../../../clinicadl/tsvtools/get_labels/get_labels.py#L177)

Removes sessions for which the diagnosis is not in the list provided

Parameters
----------
bids_df: DataFrame
    Columns must includes ['participant_id', 'session_id', 'diagnosis']
diagnosis_list: list of str
    List of diagnoses that will be removed

Returns
-------
output_df: DataFrame
    Cleaned copy of the input bids_df

#### Signature

```python
def diagnosis_removal(bids_df: pd.DataFrame, diagnosis_list: List[str]) -> pd.DataFrame:
    ...
```



## get_labels

[Show source in get_labels.py:242](../../../../clinicadl/tsvtools/get_labels/get_labels.py#L242)

Writes one TSV file based on merged_tsv and missing_mods.

Parameters
----------
bids_directory: str (path)
    Path to the folder containing the dataset in a BIDS hierarchy.
diagnoses: List of str
    Labels that must be extracted from merged_tsv.
modality: str
    Modality to select sessions. Sessions which do not include the modality will be excluded.
restriction_path: str (path)
    Path to a tsv containing the sessions that can be included.
variables_of_interest: List of str
    Columns that should be kept in the output tsv files.
remove_smc: bool
    If True SMC participants are removed from the lists.
caps_directory: str (path)
    Path to a folder of a older of a CAPS compliant dataset
merged_tsv: str (path)
    Path to the output of clinica iotools merge-tsv if already exists
missing_mods: str (path)
    Path to the output directory of clinica iotools check-missing-modalities if already exists
remove_unique_session: bool
    If True, subjects with only one session are removed.
output_dir: str (path)
    Path to the directory where the output labels.tsv will be stored.

#### Signature

```python
def get_labels(
    bids_directory: Path,
    diagnoses: List[str],
    modality: str = "t1w",
    restriction_path: Path = None,
    variables_of_interest: List[str] = None,
    remove_smc: bool = True,
    merged_tsv: Path = None,
    missing_mods: Path = None,
    remove_unique_session: bool = False,
    output_dir: Path = None,
):
    ...
```



## infer_or_drop_diagnosis

[Show source in get_labels.py:34](../../../../clinicadl/tsvtools/get_labels/get_labels.py#L34)

Deduce the diagnosis when missing from previous and following sessions of the subject. If not identical, the session
is dropped. Sessions with no diagnosis are also dropped when there are the last sessions of the follow-up.

Parameters
----------
bids_df: DataFrame
    Columns including ['participant_id', 'session_id', 'diagnosis']

Returns
-------
bids_copy_df: DataFrame
    Cleaned copy of the input bids_df

#### Signature

```python
def infer_or_drop_diagnosis(bids_df: pd.DataFrame) -> pd.DataFrame:
    ...
```



## mod_selection

[Show source in get_labels.py:106](../../../../clinicadl/tsvtools/get_labels/get_labels.py#L106)

Select only sessions for which the modality is present

Parameters
----------
bids_df: DataFrame
    Columns include ['participant_id', 'session_id', 'diagnosis']
missing_mods_dict: dictionary of str and DataFrame
    DataFrames of missing modalities
mod: str
    the modality used for selection

Returns
-------
copy_bids_df: DataFrame
    Cleaned copy of the input bids_df

#### Signature

```python
def mod_selection(
    bids_df: pd.DataFrame, missing_mods_dict: Dict[str, pd.DataFrame], mod: str = "t1w"
) -> pd.DataFrame:
    ...
```



## remove_unique_session

[Show source in get_labels.py:146](../../../../clinicadl/tsvtools/get_labels/get_labels.py#L146)

A method to get the subgroup for each sessions depending on their stability on the time horizon

Parameters
----------
bids_df: DataFrame
    Columns include ['participant_id', 'session_id', 'diagnosis']

Returns
-------
bids_copy_df: DataFrame
    Cleaned copy of the input bids_df

#### Signature

```python
def remove_unique_session(bids_df: pd.DataFrame) -> pd.DataFrame:
    ...
```