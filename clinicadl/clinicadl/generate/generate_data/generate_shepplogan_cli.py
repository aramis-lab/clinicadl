import click

from clinicadl.utils import cli_param

cmd_name = "shepplogan"


@click.command(name=cmd_name)
@cli_param.argument.generated_caps
@cli_param.option.n_subjects
@click.option(
    "--image_size",
    help="Size in pixels of the squared images.",
    type=int,
    default=128,
)
@click.option(
    "--CN_subtypes_distribution",
    "-Csd",
    type=float,
    multiple=True,
    default=(1.0, 0.0, 0.0),
    help="Probability of each subtype to be drawn in CN label.",
)
@click.option(
    "--AD_subtypes_distribution",
    "-Asd",
    type=float,
    multiple=True,
    default=(0.05, 0.85, 0.10),
    help="Probability of each subtype to be drawn in AD label.",
)
@click.option(
    "--smoothing/--no-smoothing",
    default=False,
    help="Adds random smoothing to generated data.",
)
def cli(generated_caps, img_size, AD_subtypes_distribution, CN_subtypes_distribution, n_subjects, smoothing):
    """
    Generate a dataset of 2D images at OUTPUT_CAPS_DIRECTORY including 
    3 subtypes based on Shepp Logan phantom.
    """
    from .generate import generate_shepplogan_dataset
    labels_distribution = {
            "AD": AD_subtypes_distribution,
            "CN": CN_subtypes_distribution,
        }
    generate_shepplogan_dataset(
        output_dir=generated_caps,
        img_size=img_size,
        labels_distribution=labels_distribution,
        samples=n_subjects,
        smoothing=smoothing
    )

if __name__ == "__main__":
    cli()
