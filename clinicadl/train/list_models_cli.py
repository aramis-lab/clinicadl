import click


@click.command(name="list_models")
@click.option(
    "-a",
    "--architecture",
    type=str,
    help="Basic informations about the chosen architecture",
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
    help="Architecture of the chosen model to display.",
)
def cli(
    architecture,
    input_size,
    model_layers,
):
    """Show the list of available models in ClinicaDL.
    If you choose a specific model with architecture, you will get all the basic information of the model.
    If you add the flag -model_layers, this pipeline will show the whole model layers.
    If you choose a specific shape with -input_size, it will show the whole model layers with your chosen input size.
    """
    from .train_utils import get_model_list

    get_model_list(architecture, input_size, model_layers)


if __name__ == "__main__":
    cli()
