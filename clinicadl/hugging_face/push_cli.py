import click

from clinicadl.utils import cli_param


@click.command(name="push", no_args_is_help=True)
@click.argument(
    "organization",
    type=str,
    default=None,
    # help="Path to your repo on the Hugging Face hub.",
)
@cli_param.argument.input_maps
@click.argument(
    "hf_maps_directory",
    type=str,
    default=None,
    # help="Name of the model you want to save.",
)
@click.option(
    "--dataset",
    type=str,
    default=None,
    multiple=True,
    help="Name of the dataset used for training.",
)
def cli(
    organization,
    input_maps_directory,
    hf_maps_directory,
    dataset,
):
    from .hugging_face import push_to_hf_hub

    push_to_hf_hub(
        hf_hub_path=organization,
        maps_dir=input_maps_directory,
        model_name=hf_maps_directory,
        dataset=dataset,
    )


if __name__ == "__main__":
    cli()
