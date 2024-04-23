import click

from clinicadl.generate import generate_param


@click.command(name="hypometabolic", no_args_is_help=True)
@generate_param.argument.caps_directory
@generate_param.argument.generated_caps_directory
@generate_param.option.n_proc
@generate_param.option.participants_tsv
@generate_param.option.n_subjects
@generate_param.option.use_uncropped_image
@generate_param.option_hypometabolic.sigma
@generate_param.option_hypometabolic.anomaly_degree
@generate_param.option_hypometabolic.pathology
def cli(caps_directory, generated_caps_directory, **kwargs):
    """Generation of trivial dataset with addition of synthetic brain atrophy.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_hypometabolic_dataset

    generate_hypometabolic_dataset(
        caps_directory=caps_directory,
        tsv_path=kwargs["participants_tsv"],
        preprocessing="pet-linear",
        output_dir=generated_caps_directory,
        n_subjects=kwargs["n_subjects"],
        n_proc=kwargs["n_proc"],
        pathology=kwargs["pathology"],
        anomaly_degree=kwargs["anomaly_degree"],
        sigma=kwargs["sigma"],
        uncropped_image=kwargs["use_uncropped_image"],
    )


if __name__ == "__main__":
    cli()
