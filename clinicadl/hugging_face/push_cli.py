from pathlib import Path

import click

from clinicadl.utils import cli_param
from clinicadl.utils.maps_manager import MapsManager


@click.command(name="push", no_args_is_help=True)
@cli_param.argument.input_maps
@click.argument(
    "hf_hub_path",
    type=str,
    default=None,
    # help="Path to your repo on the Hugging Face hub.",
)
@click.argument(
    "maps_name",
    type=str,
    default="maps",
    # help="Path to your repo on the Hugging Face hub.",
)
@click.option(
    "--model_name",
    type=str,
    default=None,
    help="Name of the model you want to save.",
)
@click.option(
    "--dataset",
    type=str,
    default=None,
    multiple=True,
    help="Name of the dataset used for training.",
)
def cli(
    input_maps_directory,
    hf_hub_path,
    maps_name,
    model_name,
    dataset,
):
    from .hugging_face import push_to_hf_hub

    push_to_hf_hub(
        hf_hub_path=hf_hub_path,
        maps_name=maps_name,
        maps_dir=input_maps_directory,
        model_name=model_name,
        dataset=dataset,
    )


if __name__ == "__main__":
    cli()
