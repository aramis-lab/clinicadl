from pathlib import Path

import click

from clinicadl.utils import cli_param


@click.command(name="t1-volume", no_args_is_help=True)
@cli_param.argument.caps_directory
@click.argument(
    "output_directory",
    type=click.Path(path_type=Path),
)
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
