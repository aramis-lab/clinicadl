import click

from clinicadl.utils import cli_param

cmd_name = "t1-volume"


@click.command(name=cmd_name)
@cli_param.argument.caps_directory
@click.argument(
    "output_tsv",
    type=str,
)
@click.argument(
    "group_label",
    type=str,
)
def cli(
    caps_directory,
    output_tsv,
    group_label,
):
    """Performs quality check on t1-volume pipeline.

    CAPS_DIRECTORY is the CAPS folder where t1-volume outputs are stored.

    OUTPUT_TSV is the path to the tsv file where results will be saved.
    """
    from .quality_check import quality_check as volume_qc

    volume_qc(
        caps_directory,
        output_tsv,
        group_label,
    )
