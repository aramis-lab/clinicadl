import click

from clinicadl.generate import generate_param


@click.command(name="shepplogan", no_args_is_help=True)
@generate_param.argument.generated_caps_directory
@generate_param.option.n_subjects
@generate_param.option.n_proc
@generate_param.option_shepplogan.extract_json
@generate_param.option_shepplogan.image_size
@generate_param.option_shepplogan.cn_subtypes_distribution
@generate_param.option_shepplogan.ad_subtypes_distribution
@generate_param.option_shepplogan.smoothing
def cli(generated_caps_directory, **kwargs):
    """Random generation of 2D Shepp-Logan phantoms.

    Generate a dataset of 2D images at GENERATED_CAPS_DIRECTORY including
    3 subtypes based on Shepp-Logan phantom.
    """
    from .generate import generate_shepplogan_dataset

    labels_distribution = {
        "AD": kwargs["ad_subtypes_distribution"],
        "CN": kwargs["cn_subtypes_distribution"],
    }
    generate_shepplogan_dataset(
        output_dir=generated_caps_directory,
        img_size=kwargs["image_size"],
        n_proc=kwargs["n_proc"],
        labels_distribution=labels_distribution,
        extract_json=kwargs["extract_json"],
        samples=kwargs["n_subjects"],
        smoothing=kwargs["smoothing"],
    )


if __name__ == "__main__":
    cli()
