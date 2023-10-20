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
@click.option(
    "--model_name",
    type=str,
    default=None,
    help="Name of the model you want to save.",
)
def cli(
    input_maps_directory,
    hf_hub_path,
    model_name,
):
    # maps_manager = MapsManager(input_maps_directory, verbose=None)
    from .hugging_face import push_to_hf_hub

    # network = maps_manager.get_model()
    # model_name = maps_manager.parameters["architecture"]

    push_to_hf_hub(
        hf_hub_path=hf_hub_path,
        maps_dir=input_maps_directory,
        model_name=model_name,
    )


if __name__ == "__main__":
    cli()
