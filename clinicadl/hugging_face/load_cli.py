from pathlib import Path

import click

from clinicadl.utils import cli_param


@click.command(name="load", no_args_is_help=True)
@cli_param.argument.input_maps
@click.argument(
    "hf_hub_path",
    # help="Path to your repo on the Hugging Face hub.",
    type=str,
    default=None,
)
def cli(
    input_maps_directory,
    hf_hub_path,
):

    from .hugging_face import load_from_hf_hub

    cls = input_maps_directory
    allow_pickle = ""

    load_from_hf_hub(
        cls=cls,
        hf_hub_path=hf_hub_path,
        allow_pickle=allow_pickle,
    )


if __name__ == "__main__":
    cli()
