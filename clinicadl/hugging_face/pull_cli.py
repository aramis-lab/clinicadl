from pathlib import Path

import click

from clinicadl.config import arguments


@click.command(name="pull", no_args_is_help=True)
@click.argument(
    "hf_hub_path",
    type=str,
    default=None,
)
@click.argument(
    "maps_name",
    type=str,
    default="maps",
)
@arguments.output_maps
def cli(hf_hub_path, maps_name, output_maps_directory):
    from .hugging_face import load_from_hf_hub

    load_from_hf_hub(
        output_maps=output_maps_directory,
        hf_hub_path=hf_hub_path,
        maps_name=maps_name,
    )


if __name__ == "__main__":
    cli()
