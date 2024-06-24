import click

from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    data,
    dataloader,
    extraction,
    preprocessing,
)
from clinicadl.utils.enum import ExtractionMethod, Preprocessing


@click.command(name="pet-linear", no_args_is_help=True)
@arguments.caps_directory
@arguments.results_tsv
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.use_uncropped_image
@data.participants_tsv
@dataloader.n_proc
@click.option(
    "--threshold",
    type=float,
    default=0.9,
    show_default=True,
    help="The threshold on the output probability to decide if the image passed or failed.",
)
def cli(
    caps_directory,
    results_tsv,
    tracer,
    suvr_reference_region,
    use_uncropped_image,
    participants_tsv,
    threshold,
    n_proc,
):
    """Performs quality check on pet-linear pipeline.

    CAPS_DIRECTORY is the CAPS folder where pet-linear outputs are stored.

    OUTPUT_TSV is the path to TSV output file.

    TRACER is the label given to the PET acquisition, specifying the tracer used (trc-<tracer>).

    SUVR_REFERENCE_REGION is the reference region used to perform intensity normalization {pons|cerebellumPons|pons2|cerebellumPons2}.
    """
    from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig

    from .....quality_check.pet_linear.quality_check import (
        quality_check as pet_linear_qc,
    )

    config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        caps_directory=caps_directory,
        tracer=tracer,
        suvr_reference_region=suvr_reference_region,
        use_uncropped_image=use_uncropped_image,
        data_tsv=participants_tsv,
        n_proc=n_proc,
        preprocessing_type=Preprocessing.PET_LINEAR,
        extraction=ExtractionMethod.IMAGE,
    )

    pet_linear_qc(config, output_tsv=results_tsv, threshold=threshold)
