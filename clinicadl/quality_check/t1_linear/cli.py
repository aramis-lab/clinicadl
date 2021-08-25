import click

from clinicadl.utils import cli_param


@click.command(name="t1-linear")
@cli_param.argument.caps_directory
@click.argument(
    "output_tsv",
    type=str,
)
@cli_param.option.participant_list
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="The threshold on the output probability to decide if the image "
    "passed or failed.",
)
@cli_param.option.batch_size
@cli_param.option.n_proc
@cli_param.option.use_gpu
def cli(
    caps_directory,
    output_tsv,
    participants_tsv,
    threshold,
    batch_size,
    n_proc,
    gpu,
):
    """Performs quality check on t1-linear pipeline.

    CAPS_DIRECTORY is the CAPS folder where t1-linear outputs are stored.

    OUTPUT_TSV is the path to the tsv file where results will be saved.
    """
    from clinicadl.utils.cmdline_utils import check_gpu

    if gpu:
        check_gpu()

    from .quality_check import quality_check as linear_qc

    linear_qc(
        caps_directory,
        output_tsv,
        tsv_path=participants_tsv,
        threshold=threshold,
        batch_size=batch_size,
        num_workers=n_proc,
        gpu=gpu,
    )
