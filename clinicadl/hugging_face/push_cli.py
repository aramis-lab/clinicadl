import click

from clinicadl.utils import cli_param


@click.command(name="push", no_args_is_help=True)
@click.argument(
    "organization",
    type=str,
    default=None,
)
@cli_param.argument.input_maps
@click.argument(
    "hf_maps_directory",
    type=str,
    default=None,
)
def cli(
    organization,
    input_maps_directory,
    hf_maps_directory,
):
    from .hugging_face import push_to_hf_hub

    push_to_hf_hub(
        hf_hub_path=organization,
        maps_dir=input_maps_directory,
        model_name=hf_maps_directory,
    )


if __name__ == "__main__":
    cli()
