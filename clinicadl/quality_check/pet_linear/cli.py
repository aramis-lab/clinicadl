import click

from clinicadl.utils import cli_param


@click.command(name="pet-linear", no_args_is_help=True)
@cli_param.argument.caps_directory
@click.argument(
    "output_directory",
    type=str,
)
@click.argument(
    "acq_label",
    type=str,
)
@click.argument(
    "ref_region",
    type=str,
)
@click.option(
    "--threshold",
    type=float,
    default=0.9,
    show_default=True,
    help="The threshold on the output probability to decide if the image "
    "passed or failed.",
)
def cli(
    caps_directory,
    output_directory,
    acq_label,
    ref_region,
    threshold,
):
    """Performs quality check on t1-volume pipeline.

    CAPS_DIRECTORY is the CAPS folder where t1-volume outputs are stored.

    OUTPUT_DIRECTORY is the path to the directory in which TSV files will be written.

    GROUP_LABEL is the group associated to the gray matter DARTEL template in CAPS_DIRECTORY.
    """
    from .quality_check import quality_check as pet_linear_qc

    pet_linear_qc(
        caps_directory,
        output_directory,
        acq_label,
        ref_region,
        threshold,
    )
