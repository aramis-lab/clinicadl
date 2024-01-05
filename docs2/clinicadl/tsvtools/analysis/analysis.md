# Analysis

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Analysis](./index.md#analysis) /
Analysis

> Auto-generated documentation for [clinicadl.tsvtools.analysis.analysis](../../../../clinicadl/tsvtools/analysis/analysis.py) module.

- [Analysis](#analysis)
  - [demographics_analysis](#demographics_analysis)

## demographics_analysis

[Show source in analysis.py:23](../../../../clinicadl/tsvtools/analysis/analysis.py#L23)

Produces a tsv file with rows corresponding to the labels defined by the diagnoses list,
and the columns being demographic statistics.

Writes one tsv file at results_tsv containing the demographic analysis of the tsv files in data_tsv.

Parameters
----------
merged_tsv: str (path)
    Path to the file obtained by the command clinica iotools merge-tsv.
data_tsv: str (path)
    Path to the folder containing data extracted by clinicadl tsvtool get-labels.
results_tsv: str (path)
    Path to the output tsv file (filename included).
diagnoses: list of str
    Labels selected for the demographic analysis.

#### Signature

```python
def demographics_analysis(
    merged_tsv: Path, data_tsv: Path, results_tsv: Path, diagnoses
):
    ...
```