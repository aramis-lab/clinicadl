import click

from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    data,
    dataloader,
    modality,
    preprocessing,
)


@click.command(name="pet-linear", no_args_is_help=True)
@arguments.caps_directory
@arguments.results_tsv
@modality.tracer
@modality.suvr_reference_region
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
    from .quality_check import quality_check as pet_linear_qc

    pet_linear_qc(
        caps_directory,
        output_tsv=results_tsv,
        tracer=tracer,
        ref_region=suvr_reference_region,
        use_uncropped_image=use_uncropped_image,
        participants_tsv=participants_tsv,
        threshold=threshold,
        n_proc=n_proc,
    )
