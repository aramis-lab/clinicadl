"""Common CLI arguments used by ClinicaDL pipelines."""
import click

caps_directory = click.argument("caps_directory", type=click.Path(exists=True))
input_maps = click.argument("input_maps_directory", type=click.Path(exists=True))
output_maps = click.argument("output_maps_directory", type=click.Path())

# TSV TOOLS
merged_tsv = click.argument("merged_tsv", type=click.Path(exists=True))

formatted_data_directory = click.argument(
    "formatted_data_directory", type=click.Path(exists=True)
)
missing_mods_directory = click.argument(
    "missing_mods_directory", type=click.Path(exists=True)
)

dataset = click.argument("dataset", type=click.Choice(["AIBL", "OASIS"]))
results_directory = click.argument("results_directory", type=click.Path())

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
