import click

from clinicadl.utils import cli_param


@click.command(name="shepplogan", no_args_is_help=True)
@cli_param.argument.generated_caps
@cli_param.option.n_subjects
@cli_param.option.n_proc
@cli_param.option.extract_json
@click.option(
    "--image_size",
    help="Size in pixels of the squared images.",
    type=int,
    default=128,
)
@click.option(
    "--cn_subtypes_distribution",
    "-csd",
    type=float,
    multiple=3,
    default=(1.0, 0.0, 0.0),
    help="Probability of each subtype to be drawn in CN label.",
)
@click.option(
    "--ad_subtypes_distribution",
    "-asd",
    type=float,
    multiple=3,
    default=(0.05, 0.85, 0.10),
    help="Probability of each subtype to be drawn in AD label.",
)
@click.option(
    "--smoothing/--no-smoothing",
    default=False,
    help="Adds random smoothing to generated data.",
)
def cli(
    generated_caps_directory,
    image_size,
    n_proc,
    extract_json,
    ad_subtypes_distribution,
    cn_subtypes_distribution,
    n_subjects,
    smoothing,
):
    """Random generation of 2D Shepp-Logan phantoms.

    Generate a dataset of 2D images at GENERATED_CAPS_DIRECTORY including
    3 subtypes based on Shepp-Logan phantom.
    """
    from .generate import generate_shepplogan_dataset

    labels_distribution = {
        "AD": ad_subtypes_distribution,
        "CN": cn_subtypes_distribution,
    }
    generate_shepplogan_dataset(
        output_dir=generated_caps_directory,
        img_size=image_size,
        n_proc=n_proc,
        labels_distribution=labels_distribution,
        extract_json=extract_json,
        samples=n_subjects,
        smoothing=smoothing,
    )


if __name__ == "__main__":
    cli()
