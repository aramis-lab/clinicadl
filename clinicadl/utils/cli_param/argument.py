"""Common CLI arguments used by ClinicaDL pipelines."""
import click

bids_directory = click.argument("bids_directory", type=click.Path(exists=True))
caps_directory = click.argument("caps_directory", type=click.Path(exists=True))
input_maps = click.argument("input_maps_directory", type=click.Path(exists=True))
output_maps = click.argument("output_maps_directory", type=click.Path())
output_tsv = click.argument("output_tsv", type=str)

# ANALYSIS
merged_tsv = click.argument("merged_tsv", type=click.Path(exists=True))

# TSV TOOLS
formatted_data_tsv = click.argument("formatted_data_tsv", type=click.Path(exists=True))

dataset = click.argument("dataset", type=click.Choice(["AIBL", "OASIS"]))
results_tsv = click.argument("results_tsv", type=click.Path())

# GENERATE
generated_caps = click.argument("generated_caps_directory", type=click.Path())

# PREDICT
data_group = click.argument("data_group", type=str)

# TRAIN
preprocessing_json = click.argument("preprocessing_json", type=str)

# EXTRACT
modality = click.argument(
    "modality",
    type=click.Choice(["t1-linear", "pet-linear", "custom"]),
)
