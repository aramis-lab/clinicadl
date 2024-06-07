import click

from clinicadl.commandline import arguments


@click.command(name="t1-volume", no_args_is_help=True)
@arguments.caps_directory
@arguments.output_directory
@click.argument(
    "group_label",
    type=str,
)
def cli(
    caps_directory,
    output_directory,
    group_label,
):
    """Performs quality check on t1-volume pipeline.

    CAPS_DIRECTORY is the CAPS folder where t1-volume outputs are stored.

    OUTPUT_DIRECTORY is the path to the directory in which TSV files will be written.

    GROUP_LABEL is the group associated to the gray matter DARTEL template in CAPS_DIRECTORY.
    """
    from .quality_check import quality_check as volume_qc

    volume_qc(
        caps_directory,
        output_directory,
        group_label,
    )
