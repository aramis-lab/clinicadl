from pathlib import Path

import click

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import computational, data, dataloader
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.enum import ExtractionMethod, Preprocessing


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
    from clinicadl.quality_check.t1_linear.quality_check import (
        quality_check as linear_qc,
    )

    computational_config = ComputationalConfig(amp=amp, gpu=gpu)
    config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        caps_directory=caps_directory,
        extraction=ExtractionMethod.IMAGE,
        preprocessing_type=Preprocessing.T1_LINEAR,
        preprocessing=Preprocessing.T1_LINEAR,
        use_uncropped_image=use_uncropped_image,
        data_tsv=participants_tsv,
        n_proc=n_proc,
        batch_size=batch_size,
        use_tensor=use_tensor,
    )

    linear_qc(
        output_path=results_tsv,
        threshold=threshold,
        network=network,
        use_tensor=use_tensor,
        config=config,
        computational_config=computational_config,
    )
