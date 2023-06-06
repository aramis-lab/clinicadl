from pathlib import Path

import click

from clinicadl.utils import cli_param


@click.command(name="trivial", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.preprocessing
@cli_param.option.participant_list
@cli_param.option.n_subjects
@cli_param.option.n_proc
@click.option(
    "--mask_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to the extracted masks to generate the two labels. "
    "Default will try to download masks and store them at '~/.cache/clinicadl'.",
)
@click.option(
    "--atrophy_percent",
    type=float,
    default=60.0,
    help="Percentage of atrophy applied.",
)
@cli_param.option.use_uncropped_image
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
def cli(
    caps_directory,
    generated_caps_directory,
    preprocessing,
    participants_tsv,
    n_subjects,
    n_proc,
    mask_path,
    atrophy_percent,
    use_uncropped_image,
    tracer,
    suvr_reference_region,
):
    """Generation of trivial dataset with addition of synthetic brain atrophy.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_trivial_dataset

    generate_trivial_dataset(
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        preprocessing=preprocessing,
        output_dir=generated_caps_directory,
        n_subjects=n_subjects,
        n_proc=n_proc,
        mask_path=mask_path,
        atrophy_percent=atrophy_percent,
        uncropped_image=use_uncropped_image,
        tracer=tracer,
        suvr_reference_region=suvr_reference_region,
    )


if __name__ == "__main__":
    cli()
