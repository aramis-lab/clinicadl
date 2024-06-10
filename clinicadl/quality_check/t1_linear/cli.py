from pathlib import Path

import click

from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import computational, data, dataloader


@click.command(name="t1-linear", no_args_is_help=True)
@arguments.caps_directory
@arguments.results_tsv
@data.participants_tsv
@dataloader.batch_size
@dataloader.n_proc
@computational.gpu
@computational.amp
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="The threshold on the output probability to decide if the image "
    "passed or failed.",
)
@click.option(
    "--network",
    default="darq",
    type=click.Choice(["darq", "deep_qc", "sq101"]),
    help="is the architecture chosen for the network (to chose between darq, sq101 and deep_qc",
)
@click.option(
    "--use_tensor",
    type=bool,
    default=False,
    is_flag=True,
    help="Flag allowing the pipeline to run on the extracted tensors and not on the nifti images",
)
def cli(
    caps_directory,
    results_tsv,
    participants_tsv,
    threshold,
    batch_size,
    n_proc,
    gpu,
    amp,
    network,
    use_tensor,
    use_uncropped_image=True,
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
        output_path=results_tsv,
        tsv_path=participants_tsv,
        threshold=threshold,
        batch_size=batch_size,
        n_proc=n_proc,
        gpu=gpu,
        amp=amp,
        network=network,
        use_tensor=use_tensor,
        use_uncropped_image=use_uncropped_image,
    )
