from pathlib import Path

import click

from clinicadl.generate import generate_param
from clinicadl.generate.generate import generate_artifacts_dataset
from clinicadl.generate.generate_config import GenerateArtifactsConfig

config = GenerateArtifactsConfig.model_fields


@click.command(name="artifacts", no_args_is_help=True)
@generate_param.argument.caps_directory
@generate_param.argument.generated_caps_directory
@generate_param.option.n_proc
@generate_param.option.preprocessing
@generate_param.option.participants_tsv
@generate_param.option.use_uncropped_image
@generate_param.option.tracer
@generate_param.option.suvr_reference_region
@generate_param.option_artifacts.contrast
@generate_param.option_artifacts.motion
@generate_param.option_artifacts.noise_std
@generate_param.option_artifacts.noise
@generate_param.option_artifacts.num_transforms
@generate_param.option_artifacts.translation
@generate_param.option_artifacts.rotation
@generate_param.option_artifacts.gamma
def cli(caps_directory, generated_caps_directory, **kwargs):
    """Generation of trivial dataset with addition of synthetic artifacts.
    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.
    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """

    generate_artifacts_dataset(
        caps_directory=caps_directory,
        tsv_path=kwargs["participants_tsv"],
        preprocessing=kwargs["preprocessing"],
        output_dir=generated_caps_directory,
        uncropped_image=kwargs["use_uncropped_image"],
        tracer=kwargs["tracer"],
        suvr_reference_region=kwargs["suvr_reference_region"],
        contrast=kwargs["contrast"],
        gamma=kwargs["gamma"],
        motion=kwargs["motion"],
        translation=kwargs["translation"],
        rotation=kwargs["rotation"],
        num_transforms=kwargs["num_transforms"],
        noise=kwargs["noise"],
        noise_std=kwargs["noise_std"],
        n_proc=kwargs["n_proc"],
    )


if __name__ == "__main__":
    cli()
