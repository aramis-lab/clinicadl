import click

from clinicadl.utils import cli_param


@click.command(name="random")
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.preprocessing
@cli_param.option.participant_list
@cli_param.option.n_subjects
@click.option(
    "--mean",
    type=float,
    default=0,
    help="Mean value of the gaussian noise added to synthetic images.",
)
@click.option(
    "--sigma",
    type=float,
    default=0.5,
    help="Standard deviation of the gaussian noise added to synthetic images.",
)
@cli_param.option.use_uncropped_image
@cli_param.option.acq_label
@cli_param.option.suvr_reference_region
def cli(
    caps_directory,
    generated_caps_directory,
    preprocessing,
    participants_tsv,
    n_subjects,
    mean,
    sigma,
    use_uncropped_image,
    acq_label,
    suvr_reference_region,
):
    """Addition of random gaussian noise to brain images.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the random dataset will be saved.
    """
    from .generate import generate_random_dataset

    generate_random_dataset(
        caps_directory=caps_directory,
        preprocessing=preprocessing,
        tsv_path=participants_tsv,
        output_dir=generated_caps_directory,
        n_subjects=n_subjects,
        mean=mean,
        sigma=sigma,
        uncropped_image=use_uncropped_image,
        acq_label=acq_label,
        suvr_reference_region=suvr_reference_region,
    )


if __name__ == "__main__":
    cli()
