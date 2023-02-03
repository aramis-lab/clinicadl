import click

from clinicadl.utils import cli_param


@click.command(name="pet-linear", no_args_is_help=True)
@cli_param.argument.caps_directory
@click.argument(
    "output_tsv",
    type=str,
)
@click.argument(
    "acq_label",
    type=str,
    help="is the label given to the PET acquisition, specifying the tracer used (trc-<acq_label>).",
)
@click.argument(
    "suvr_reference_region",
    type=str,
    help="is the reference region used to perform intensity normalization {pons|cerebellumPons|pons2|cerebellumPons2}.",
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
@cli_param.option.n_proc
def cli(
    caps_directory,
    output_tsv,
    acq_label,
    ref_region,
    participants_tsv,
    threshold,
    n_proc,
):
    """Performs quality check on pet-linear pipeline.

    CAPS_DIRECTORY is the CAPS folder where pet-linear outputs are stored.

    OUTPUT_TSV is the path to TSV output file.

    ACQ_LABEL is the label given to the PET acquisition, specifying the tracer used (trc-<acq_label>).

    SUVR_REFERENCE_REGION is the reference region used to perform intensity normalization {pons|cerebellumPons|pons2|cerebellumPons2}.
    """
    from .quality_check import quality_check as pet_linear_qc

    pet_linear_qc(
        caps_directory,
        output_tsv,
        acq_label,
        ref_region,
        participants_tsv=participants_tsv,
        threshold=threshold,
        n_proc=n_proc,
    )
