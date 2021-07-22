"""Common CLI arguments used by ClinicaDL pipelines."""
import click


caps_directory = click.argument(
    "input_caps_directory", type=click.Path(exists=True)
)
input_maps = click.argument(
    "input_maps_directory", type=click.Path(exists=True)
)
output_maps = click.argument(
    "output_maps_directory", type=click.Path()
)

## TSV TOOLS
merged_tsv = click.argument(
    "merged_tsv", type=click.File()
)

formatted_data_directory = click.argument(
    "formatted_data_directory", type=click.Path(exists=True)
)

missing_mods_directory = click.argument(
    "missing_mods_directory", type=click.Path(exists=True)
)

dataset = click.argument(
    "dataset", type=click.Choice(["AIBL", "OASIS"])
)
results_directory = click.argument(
    "results_directory", type=click.Path()
)

## GENERATE

participant_list = click.argument(
    "participants_tsv", type=click.File()
)
generated_caps = click.argument(
    "output_caps_directory", type=click.Path()
)

## INFER

inference_prefix = click.argument(
    "output_prefix", type=str
)

## TRAIN

preprocessing_json = click.argument(
    "preprocessing_json", type=click.File()
)