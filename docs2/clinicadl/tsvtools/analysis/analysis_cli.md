# Analysis Cli

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Analysis](./index.md#analysis) /
Analysis Cli

> Auto-generated documentation for [clinicadl.tsvtools.analysis.analysis_cli](../../../../clinicadl/tsvtools/analysis/analysis_cli.py) module.

- [Analysis Cli](#analysis-cli)
  - [cli](#cli)

## cli

[Show source in analysis_cli.py:6](../../../../clinicadl/tsvtools/analysis/analysis_cli.py#L6)

Demographic analysis of the extracted labels.

MERGED_TSV is the output of `clinica iotools merge-tsv`.

DATA_TSV is the output of `clinicadl tsvtools get-labels`.

Results are stored in RESULTS_TSV.

#### Signature

```python
@click.command(name="analysis", no_args_is_help=True)
@cli_param.argument.merged_tsv
@cli_param.argument.data_tsv
@cli_param.argument.results_tsv
@cli_param.option.diagnoses
def cli(merged_tsv, data_tsv, results_tsv, diagnoses):
    ...
```