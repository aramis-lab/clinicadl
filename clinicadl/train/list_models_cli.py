import click


@click.command(name="list_models")
@click.option(
    "-a",
    "--architecture",
    type=str,
    help="Name of the network for which information will be displayed.",
)
@click.option(
    "-i",
    "--input_size",
    type=str,
    help="Size of the input image in the shape C@HxW if the image is 2D or C@DxHxW if the image is 3D.",
)
@click.option(
    "-m",
    "--model_layers",
    type=bool,
    default=False,
    is_flag=True,
    help="Display the detailed Pytorch architecture.",
)
def cli(
    architecture,
    input_size,
    model_layers,
):
    """Show the list of available models in ClinicaDL."""
    from .train_utils import get_model_list

    get_model_list(architecture, input_size, model_layers)


if __name__ == "__main__":
    cli()
