# Predict Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Predict](./index.md#predict) /
Predict Cli

> Auto-generated documentation for [clinicadl.predict.predict_cli](../../../clinicadl/predict/predict_cli.py) module.

- [Predict Cli](#predict-cli)
  - [cli](#cli)

## cli

[Show source in predict_cli.py:8](../../../clinicadl/predict/predict_cli.py#L8)

Infer the outputs of a trained model on a test set.

INPUT_MAPS_DIRECTORY is the MAPS folder from where the model used for prediction will be loaded.

DATA_GROUP is the name of the subjects and sessions list used for the interpretation.

#### Signature

```python
@click.command(name="predict", no_args_is_help=True)
@cli_param.argument.input_maps
@cli_param.argument.data_group
@click.option(
    "--caps_directory",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Data using CAPS structure, if different from the one used during network"
        " training."
    ),
)
@click.option(
    "--participants_tsv",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to the file with subjects/sessions to process, if different from the one"
        " used during network training.\n    If it includes the filename will load the"
        " TSV file directly.\n    Else will load the baseline TSV files of wanted"
        " diagnoses produced by `tsvtool split`."
    ),
)
@click.option(
    "--use_labels/--no_labels",
    default=True,
    help=(
        "Set this option to --no_labels if your dataset does not contain ground truth"
        " labels."
    ),
)
@click.option(
    "--selection_metrics",
    "-sm",
    default=["loss"],
    multiple=True,
    help=(
        "Allow to select a list of models based on their selection metric. Default"
        " will\n    only infer the result of the best model selected on loss."
    ),
)
@click.option(
    "--multi_cohort",
    type=bool,
    default=False,
    is_flag=True,
    help=(
        "Allow to use multiple CAPS directories.\n            In this case,"
        " CAPS_DIRECTORY and PARTICIPANTS_TSV must be paths to TSV files."
    ),
)
@click.option(
    "--diagnoses",
    "-d",
    type=str,
    multiple=True,
    help=(
        "List of diagnoses used for inference. Is used only if PARTICIPANTS_TSV leads to"
        " a folder."
    ),
)
@click.option(
    "--label",
    type=str,
    default=None,
    help=(
        "Target label used for training (if NETWORK_TASK in [`regression`,"
        " `classification`]). Default will reuse the same label as during the training"
        " task."
    ),
)
@click.option(
    "--save_tensor",
    type=bool,
    default=False,
    is_flag=True,
    help="Save the reconstruction output in the MAPS in Pytorch tensor format.",
)
@cli_param.option.save_nifti
@cli_param.option.use_gpu
@cli_param.option.n_proc
@cli_param.option.batch_size
@cli_param.option.overwrite
def cli(
    input_maps_directory,
    data_group,
    caps_directory,
    participants_tsv,
    gpu,
    n_proc,
    batch_size,
    use_labels,
    label,
    selection_metrics,
    diagnoses,
    multi_cohort,
    overwrite,
    save_tensor,
    save_nifti,
):
    ...
```