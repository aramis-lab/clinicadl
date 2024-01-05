# Split

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Split](./index.md#split) /
Split

> Auto-generated documentation for [clinicadl.tsvtools.split.split](../../../../clinicadl/tsvtools/split/split.py) module.

- [Split](#split)
  - [KStests](#kstests)
  - [create_split](#create_split)
  - [shuffle_choice](#shuffle_choice)
  - [split_diagnoses](#split_diagnoses)

## KStests

[Show source in split.py:28](../../../../clinicadl/tsvtools/split/split.py#L28)

#### Signature

```python
def KStests(train_df, test_df, threshold=0.5):
    ...
```



## create_split

[Show source in split.py:57](../../../../clinicadl/tsvtools/split/split.py#L57)

Split data at the subject-level in training and test set with equivalent age, sex and split_label distributions

Parameters
----------
diagnosis_df: DataFrame
    Columns including ['participant_id', 'session_id', 'group']
split_label: str
    Label on which the split is done (categorical variables)
n_test: float
    If > 1 number of subjects to put in the test set.
    If < 1 proportion of subjects to put in the test set.
p_age_threshold: float
    Threshold for the t-test on age.
p_sex_threshold: float
    Threshold for the chi2 test on sex.
supplementary_train_df: DataFrame
    Add data that must be included in the train set.
ignore_demographics: bool
    If True the diagnoses are split without taking into account the demographics
    distributions (age, sex).

Returns
-------
train_df: DataFrame
    Subjects in the train set
test_df: DataFrame
    Subjects in the test set

#### Signature

```python
def create_split(
    diagnosis_df,
    split_label,
    n_test,
    p_age_threshold=0.8,
    p_sex_threshold=0.8,
    supplementary_train_df=None,
    ignore_demographics=False,
):
    ...
```



## shuffle_choice

[Show source in split.py:41](../../../../clinicadl/tsvtools/split/split.py#L41)

#### Signature

```python
def shuffle_choice(df, n_shuffle=10):
    ...
```



## split_diagnoses

[Show source in split.py:181](../../../../clinicadl/tsvtools/split/split.py#L181)

Performs a single split for each label independently on the subject level.
There will be two TSV file for the train set (baseline and longitudinal),
whereas there will only be one TSV file for the test set (baseline sessions).

The age and sex distributions between the two sets must be non-significant (according to T-test and chi-square).

Parameters
----------
data_tsv: str (path)
    Path to the tsv containing data extracted by clinicadl tsvtools getlabels.
n_test: float
    If >= 1, number of subjects to put in set with name 'subset_name'.
    If < 1, proportion of subjects to put in set with name 'subset_name'.
    If 0, no training set is created and the whole dataset is considered as one set with name 'subset_name'.
subset_name: str
    Name of the subset that is complementary to train.
p_age_threshold: float
    The threshold used for the T-test on age distributions.
p_sex_threshold: float
    The threshold used for the T-test on sex distributions.
categorical_split_variable: str
    Name of a categorical variable to perform a stratified split.
ignore_demographics: bool
    If True the diagnoses are split without taking into account the demographics
    distributions (age, sex).
verbose: int
    Level of verbosity.

Informations
------------
writes three files per <label>.tsv file present in data_tsv:
    - data_tsv/train/<label>.tsv
    - data_tsv/train/<label>_baseline.tsv
    - data_tsv/<subset_name>/<label>_baseline.tsv

#### Signature

```python
def split_diagnoses(
    data_tsv: Path,
    n_test=100,
    subset_name="test",
    p_age_threshold=0.8,
    p_sex_threshold=0.8,
    categorical_split_variable=None,
    ignore_demographics=False,
    verbose=0,
    not_only_baseline=True,
    multi_diagnoses=False,
):
    ...
```