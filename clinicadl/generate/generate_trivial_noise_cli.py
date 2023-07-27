from typing import Dict, List, Optional, Tuple, Union

import click

from clinicadl.utils import cli_param


@click.command(name="trivial_noise", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.n_proc
@cli_param.option.preprocessing
@cli_param.option.participant_list
@cli_param.option.use_uncropped_image
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@click.option(
    "--noise_std",
    type=float,
    multiple=2,
    default=[5, 15],
    help="Range for noise standard deviation",
)
def cli(
    caps_directory,
    generated_caps_directory,
    preprocessing,
    participants_tsv,
    use_uncropped_image,
    tracer,
    suvr_reference_region,
    noise_std,
    n_proc,
):
    """Generation of trivial dataset with addition of synthetic noise artifacts.
    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.
    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_noise_dataset

    generate_noise_dataset(
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        preprocessing=preprocessing,
        output_dir=generated_caps_directory,
        uncropped_image=use_uncropped_image,
        tracer=tracer,
        suvr_reference_region=suvr_reference_region,
        noise_std=noise_std,
        n_proc=n_proc,
    )


if __name__ == "__main__":
    cli()
