import click

from clinicadl.utils import cli_param


@click.command(name="hypometabolic", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.participant_list
@cli_param.option.n_subjects
@cli_param.option.n_proc
@click.option(
    "--pathology",
    "-p",
    type=str,
    default="ad",
    help="Path to the extracted masks to generate the two labels. "
    "Default will try to download masks and store them at '~/.cache/clinicadl'.",
)
@click.option(
    "--pathology_percent",
    type=float,
    default=60.0,
    help="Percentage of pathology applied.",
)
@cli_param.option.use_uncropped_image
@cli_param.option.acq_label
@cli_param.option.suvr_reference_region
def cli(
    caps_directory,
    generated_caps_directory,
    participants_tsv,
    n_subjects,
    n_proc,
    pathology,
    pathology_percent,
    use_uncropped_image,
    acq_label,
    suvr_reference_region,
):
    """Generation of trivial dataset with addition of synthetic brain atrophy.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_hypometabolic_dataset

    generate_hypometabolic_dataset(
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        preprocessing="pet-linear",
        output_dir=generated_caps_directory,
        n_subjects=n_subjects,
        n_proc=n_proc,
        pathology=pathology,
        pathology_percent=pathology_percent,
        uncropped_image=use_uncropped_image,
        acq_label="18FFDG",
        suvr_reference_region="cerebellumPons2",
    )


if __name__ == "__main__":
    cli()
