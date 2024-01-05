# Kfold

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Kfold](./index.md#kfold) /
Kfold

> Auto-generated documentation for [clinicadl.tsvtools.kfold.kfold](../../../../clinicadl/tsvtools/kfold/kfold.py) module.

- [Kfold](#kfold)
  - [split_diagnoses](#split_diagnoses)
  - [write_splits](#write_splits)

## split_diagnoses

[Show source in kfold.py:91](../../../../clinicadl/tsvtools/kfold/kfold.py#L91)

Performs a k-fold split for each label independently on the subject level.
The output (the tsv file) will have two new columns :
    - split, with the number of the split the subject is in.
    - datagroup, with the name of the group (train or subset_name) the subject is in.

The train group will contain baseline and longitudinal sessions,
whereas the test group will only include the baseline sessions for each split.

Parameters
----------
data_tsv: str (path)
    Path to the tsv file extracted by clinicadl tsvtool getlabels.
n_splits: int
    Number of splits in the k-fold cross-validation.
subset_name: str
    Name of the subset that is complementary to train.
stratification: str
    Name of variable used to stratify k-fold.
merged_tsv: str
    Path to the merged.tsv file, output of clinica iotools merge-tsv.

#### Signature

```python
def split_diagnoses(
    data_tsv: Path,
    n_splits: int = 5,
    subset_name: str = None,
    stratification: str = None,
    merged_tsv: Path = None,
):
    ...
```



## write_splits

[Show source in kfold.py:18](../../../../clinicadl/tsvtools/kfold/kfold.py#L18)

Split data at the subject-level in training and test to have equivalent distributions in split_label.
Writes test and train Dataframes.

Parameters
----------
diagnosis_df: Dataframe
    Columns must include ['participant_id', 'session_id', 'diagnosis']
split_label: str
    Label on which the split is done (categorical variables)
n_splits: int
    Number of splits in the k-fold cross-validation.
subset_name: str
    Name of the subset split.
results_directory: str (path)
    Path to the results directory.

#### Signature

```python
def write_splits(
    diagnosis_df: pd.DataFrame,
    split_label: str,
    n_splits: int,
    subset_name: str,
    results_directory: Path,
):
    ...
```