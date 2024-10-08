"""Common CLI arguments used by ClinicaDL pipelines."""

from pathlib import Path

import click

# TODO trier les arguments par configclasses et voir les arguments utils et ceux qui ne le sont pas
bids_directory = click.argument(
    "bids_directory", type=click.Path(exists=True, path_type=Path)
)
caps_directory = click.argument("caps_directory", type=click.Path(path_type=Path))
input_maps = click.argument(
    "input_maps_directory", type=click.Path(exists=True, path_type=Path)
)
output_maps = click.argument("output_maps_directory", type=click.Path(path_type=Path))
results_tsv = click.argument("results_tsv", type=click.Path(path_type=Path))
data_tsv = click.argument("data_tsv", type=click.Path(exists=True, path_type=Path))
# ANALYSIS
merged_tsv = click.argument("merged_tsv", type=click.Path(exists=True, path_type=Path))

# TSV TOOLS
tsv_path = click.argument("tsv_path", type=click.Path(exists=True, path_type=Path))
old_tsv_dir = click.argument(
    "old_tsv_dir", type=click.Path(exists=True, path_type=Path)
)
new_tsv_dir = click.argument("new_tsv_dir", type=click.Path(path_type=Path))
output_directory = click.argument("output_directory", type=click.Path(path_type=Path))
dataset = click.argument("dataset", type=click.Choice(["AIBL", "OASIS"]))


# TRAIN
preprocessing_json = click.argument("preprocessing_json", type=str)

modality_bids = click.argument(
    "modality_bids",
    type=click.Choice(["t1", "pet", "flair", "dwi", "custom"]),
)
tracer = click.argument(
    "tracer",
    type=str,
)
suvr_reference_region = click.argument(
    "suvr_reference_region",
    type=str,
)
generated_caps_directory = click.argument("generated_caps_directory", type=Path)

data_group = click.argument("data_group", type=str)
config_file = click.argument(
    "config_file", type=click.Path(exists=True, path_type=Path)
)
preprocessing = click.argument(
    "preprocessing", type=click.Choice(["t1", "pet", "flair", "dwi", "custom"])
)
