# Adapt

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Adapt](./index.md#adapt) /
Adapt

> Auto-generated documentation for [clinicadl.tsvtools.adapt.adapt](../../../../clinicadl/tsvtools/adapt/adapt.py) module.

- [Adapt](#adapt)
  - [adapt](#adapt)
  - [concat_files](#concat_files)

## adapt

[Show source in adapt.py:34](../../../../clinicadl/tsvtools/adapt/adapt.py#L34)

Produces a new split/kfold directories that fit with clinicaDL 1.2.0.

Parameters
----------
old_tsv_dir: str (path)
    Path to the old directory.
new_tsv_dir: str (path)
    Path to the fnew directory.
subset_name: str
    Name of the output file of `clinicadl get-labels`.
labels_list: list of str
    list of labels (in the old way, each labels had its own TSV file).

#### Signature

```python
def adapt(old_tsv_dir: Path, new_tsv_dir: Path, subset_name="labels", labels_list="AD"):
    ...
```



## concat_files

[Show source in adapt.py:12](../../../../clinicadl/tsvtools/adapt/adapt.py#L12)

Read a file and concatenates it to the right dataframe (baseline or not).

Parameters
----------
file_path: str (path)
    Path to the TSV file (with "participant_id" and "session_id" in the columns).
df_baseline: DataFrame
    Dataframe with only baseline sessions.
df_all: DataFrame
    Dataframe with all sessions.

#### Signature

```python
def concat_files(file_path: Path, df_baseline, df_all):
    ...
```