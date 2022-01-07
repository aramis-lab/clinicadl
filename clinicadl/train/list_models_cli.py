import click


@click.command(name="list_models")
@click.option(
    "-a",
    "--architecture",
    type=str,
    help="Architecture of the chosen model to display.",
)
@click.option(
    "-in",
    "--input_size",
    type=int,
    multiple=2,
    default=(128, 128),
    show_default=True,
    help="Size of the input image.",
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
