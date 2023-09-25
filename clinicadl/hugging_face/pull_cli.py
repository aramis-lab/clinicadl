from pathlib import Path

import click

from clinicadl.utils import cli_param
from clinicadl.utils.maps_manager import MapsManager


@click.command(name="pull", no_args_is_help=True)
@cli_param.argument.output_maps
@click.argument(
    "hf_hub_path",
    type=str,
    default=None,
    # help="Path to your repo on the Hugging Face hub.",
)
def cli(
    output_maps_directory,
    hf_hub_path,
):
    from .hugging_face import load_from_hf_hub

    load_from_hf_hub(
        output_maps=output_maps_directory,
        hf_hub_path=hf_hub_path,
    )


if __name__ == "__main__":
    cli()
