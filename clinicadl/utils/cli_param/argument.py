"""Common CLI arguments used by ClinicaDL pipelines."""

from pathlib import Path

import click

bids_directory = click.argument(
    "bids_directory", type=click.Path(exists=True, path_type=Path)
)
caps_directory = click.argument("caps_directory", type=click.Path(path_type=Path))
input_maps = click.argument(
    "input_maps_directory", type=click.Path(exists=True, path_type=Path)
)
output_maps = click.argument("output_maps_directory", type=click.Path(path_type=Path))
results_tsv = click.argument("results_tsv", type=click.Path(path_type=Path))

# ANALYSIS
merged_tsv = click.argument("merged_tsv", type=click.Path(exists=True, path_type=Path))

# TSV TOOLS
data_tsv = click.argument("data_tsv", type=click.Path(exists=True, path_type=Path))
old_tsv_dir = click.argument(
    "old_tsv_dir", type=click.Path(exists=True, path_type=Path)
)
new_tsv_dir = click.argument("new_tsv_dir", type=click.Path(path_type=Path))
output_directory = click.argument("output_directory", type=click.Path(path_type=Path))
dataset = click.argument("dataset", type=click.Choice(["AIBL", "OASIS"]))

# GENERATE
generated_caps = click.argument(
    "generated_caps_directory", type=click.Path(path_type=Path)
)

# PREDICT
data_group = click.argument("data_group", type=str)

# TRAIN
preprocessing_json = click.argument("preprocessing_json", type=str)

# EXTRACT
modality = click.argument(
    "modality",
    type=click.Choice(
        [
            "t1-linear",
            "t2-linear",
            "t1-extensive",
            "dwi-dti",
            "pet-linear",
            "flair-linear",
            "custom",
        ]
    ),
)

modality_bids = click.argument(
    "modality_bids",
    type=click.Choice(["t1", "pet", "flair", "dwi", "custom"]),
)

modality_bids = click.argument(
    "modality_bids",
    type=click.Choice(["t1", "pet", "flair", "dwi", "custom"]),
)
