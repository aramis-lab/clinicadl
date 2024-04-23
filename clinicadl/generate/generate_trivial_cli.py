from pathlib import Path

import click

from clinicadl.generate import generate_param


@click.command(name="trivial", no_args_is_help=True)
@generate_param.argument.caps_directory
@generate_param.argument.generated_caps_directory
@generate_param.option.preprocessing
@generate_param.option.participants_tsv
@generate_param.option.n_subjects
@generate_param.option.n_proc
@generate_param.option.use_uncropped_image
@generate_param.option.tracer
@generate_param.option.suvr_reference_region
@generate_param.option_trivial.atrophy_percent
@generate_param.option_trivial.mask_path
def cli(caps_directory, generated_caps_directory, **kwargs):
    """Generation of trivial dataset with addition of synthetic brain atrophy.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_trivial_dataset

    generate_trivial_dataset(
        caps_directory=caps_directory,
        tsv_path=kwargs["participants_tsv"],
        preprocessing=kwargs["preprocessing"],
        output_dir=generated_caps_directory,
        n_subjects=kwargs["n_subjects"],
        n_proc=kwargs["n_proc"],
        mask_path=kwargs["mask_path"],
        atrophy_percent=kwargs["atrophy_percent"],
        uncropped_image=kwargs["use_uncropped_image"],
        tracer=kwargs["tracer"],
        suvr_reference_region=kwargs["suvr_reference_region"],
    )


if __name__ == "__main__":
    cli()
