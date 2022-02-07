import click


@click.command(name="list_models")
@click.option(
    "-a",
    "--architecture",
    type=str,
    help="Architecture of the chosen model to display.",
)
@click.option(
    "-i",
    "--input_size",
    type=str,
    default="1@128x128",
    show_default=True,
    help="Size of the input image in the shape C@HxW if the image is 2D or C@DxHxW if the image is 3D.",
)
def cli(
    architecture,
    input_size,
):
    """Show the list of available models in ClinicaDL.
    If you choose a specific model with architecture, this pipeline will show the whole model layers.
    """
    from .train_utils import get_model_list

    get_model_list(architecture, input_size)


if __name__ == "__main__":
    cli()
