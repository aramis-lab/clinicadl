import click

from clinicadl.utils import cli_param


@click.command(name="t1-linear")
@cli_param.argument.caps_directory
@click.argument(
    "output_tsv",
    type=str,
)
@click.option(
    "-tsv",
    "--subjects_sessions_tsv",
    type=str,
    default=None,
    help="TSV file containing a list of subjects with their sessions.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    help="The threshold on the output probability to decide if the image "
    "passed or failed. (default=0.5)",
)
@click.option(
    "--batch_size",
    default=1,
    type=int,
    help="Batch size used in DataLoader (default=1).",
)
@click.option(
    "-np",
    "--nproc",
    type=int,
    default=2,
    help="Number of cores used the quality check. (default=2)",
)
@cli_param.option.use_gpu
def cli(
    caps_directory,
    output_tsv,
    subjects_sessions_tsv,
    threshold,
    batch_size,
    nproc,
    gpu,
):
    """Performs quality check on t1-linear pipeline.

    CAPS_DIRECTORY is the CAPS folder where t1-linear outputs are stored.

    OUTPUT_TSV is the path to the tsv file where results will be saved.
    """
    from .quality_check import quality_check as linear_qc

    linear_qc(
        caps_directory,
        output_tsv,
        tsv_path=subjects_sessions_tsv,
        threshold=threshold,
        batch_size=batch_size,
        num_workers=nproc,
        gpu=gpu,
    )
