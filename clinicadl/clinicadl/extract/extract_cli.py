import os
from typing import Optional

import click
from clinica.utils.pet import LIST_SUVR_REFERENCE_REGIONS

from clinicadl.utils import cli_param


@click.command(name="extract")
@cli_param.argument.caps_directory
@click.argument(
    "modality",
    type=click.Choice(["t1-linear", "pet-linear", "custom"]),
)
@click.argument(
    "extract_method",
    type=click.Choice(["image", "slice", "patch", "roi"]),
)
@click.option(
    "-uui",
    "--use_uncropped_image",
    is_flag=True,
    default=False,
    help="Use the uncropped image instead of the cropped image generated by t1-linear or pet-linear.",
)
@click.option(
    "-ps",
    "--patch_size",
    default=50,
    show_default=True,
    help="Patch size if EXTRACT_METHOD is `patch`.",
)
@click.option(
    "-ss",
    "--stride_size",
    default=50,
    show_default=True,
    help="Stride size if EXTRACT_METHOD is ``patch.",
)
@click.option(
    "-sd",
    "--slice_direction",
    type=click.IntRange(0, 2),
    default=0,
    show_default=True,
    help="Slice direction if EXTRACT_METHOD is `slice`. 0: Sagittal plane, 1: Coronal plane, 2: Axial plane.",
)
@click.option(
    "-sm",
    "--slice_mode",
    type=click.Choice(["rgb", "single"]),
    default="rgb",
    show_default=True,
    help=(
        "Slice mode if EXTRACT_METHOD is `slice`. rgb: Save the slice in three identical channels, "
        "single: Save the slice in a single channel."
    ),
)
@click.option(
    "-ds",
    "--discarded_slices",
    type=int,
    default=(0, 0),
    multiple=2,
    help="""Discarded slices if EXTRACT_METHOD is `slice`. Number of slices discarded from respectively the beginning and
        the end of the MRI volume.  If only one argument is given, it will be
        used for both sides.""",
)
@click.option(
    "--roi_list",
    type=str,
    multiple=True,
    # default=(),
    help="ROI list if EXTRACT_METHOD is `roi`. List of regions to be extracted",
)
@click.option(
    "--roi_uncrop_output",
    type=bool,
    default=False,
    is_flag=True,
    help="Uncrop outputs if EXTRACT_METHOD is `roi`. Disable cropping option so the output tensors have the same size than the whole image.",
)
@click.option(
    "--roi_custom_suffix",
    "-cn",
    type=str,
    default="",
    help="""Custom suffix if EXTRACT_METHOD is `roi`. Custom suffix filename, e.g.:
        'graymatter_space-Ixi549Space_modulated-off_probability.nii.gz', or
        'segm-whitematter_probability.nii.gz' """,
)
@click.option(
    "--roi_custom_template",
    "-ct",
    type=str,
    default=None,
    help="""Custom template if EXTRACT_METHOD is `roi`.
        Name of the template used when modality is set to custom.""",
)
@click.option(
    "--roi_custom_mask_pattern",
    "-cmp",
    type=str,
    default=None,
    help="""Custom mask pattern if EXTRACT_METHOD is `roi`.
            If given will select only the masks containing the string given.
            The mask with the shortest name is taken.
            This argument is taken into account only of the modality is custom.""",
)
@click.option(
    "--acq_label",
    type=click.Choice(["av45", "fdg"]),
    help=(
        "Name of the label given to the PET acquisition, specifying  the tracer used (acq-<acq_label>). "
        "For instance it can be 'fdg' for fluorodeoxyglucose or 'av45' for florbetapir. This option only applies to "
        "the `pet-linear` pipeline."
    ),
)
@click.option(
    "-suvr",
    "--suvr_reference_region",
    type=click.Choice(LIST_SUVR_REFERENCE_REGIONS),
    help=(
        "Intensity normalization using the average PET uptake in reference regions resulting in a standardized uptake "
        "value ratio (SUVR) map. It can be cerebellumPons or cerebellumPon2 (used for amyloid tracers) or pons or "
        "pons2 (used for 18F-FDG tracers). This option only applies to `pet-linear` pipeline."
    ),
)
@click.option(
    "-cn",
    "--custom_suffix",
    default="",
    help=(
        "Suffix to append to filenames for a custom modality, for instance "
        "`graymatter_space-Ixi549Space_modulated-off_probability.nii.gz`, or "
        "`segm-whitematter_probability.nii.gz`"
    ),
)
@cli_param.option.subjects_sessions_tsv
@cli_param.option.n_proc
def cli(
    input_caps_directory: str,
    modality: str,
    extract_method: str,
    use_uncropped_image: bool = False,
    patch_size: int = 50,
    stride_size: int = 50,
    slice_direction: int = 0,
    slice_mode: str = "rgb",
    discarded_slices: int = 0,
    roi_list: list = [],
    roi_uncrop_output: bool = False,
    roi_custom_suffix: str = "",
    roi_custom_template: str = None,
    roi_custom_mask_pattern: str = None,
    acq_label: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
    custom_suffix: str = "",
    subjects_sessions_tsv: Optional[str] = None,
    nproc: Optional[int] = None,
) -> None:
    """
    Extraction of pytorch tensor from nifti images of INPUT_CAPS_DIRECTORY preprocessed
    with MODALITY clinica pipeline following EXTRACT_METHOD.
    """
    from .extract import DeepLearningPrepareData

    parameters = {
        "preprocessing": modality,
        "mode": extract_method,
        "use_uncropped_image": use_uncropped_image,
    }
    if extract_method == "slice":
        parameters["slice_direction"] = slice_direction
        parameters["slice_mode"] = slice_mode
        parameters["discarded_slices"] = discarded_slices
    elif extract_method == "patch":
        parameters["patch_size"] = patch_size
        parameters["stride_size"] = stride_size
    elif extract_method == "roi":
        parameters["roi_list"] = roi_list
        parameters["uncropped_roi"] = roi_uncrop_output
        parameters["roi_custom_suffix"] = roi_custom_suffix
        parameters["roi_custom_template"] = roi_custom_template
        parameters["roi_custom_mask_pattern"] = roi_custom_mask_pattern

    if modality == "custom":
        parameters["custom_suffix"] = custom_suffix

    if modality == "pet-linear":
        parameters["acq_label"] = acq_label
        parameters["suvr_reference_region"] = suvr_reference_region

    DeepLearningPrepareData(
        caps_directory=input_caps_directory,
        tsv_file=subjects_sessions_tsv,
        parameters=parameters,
    )


if __name__ == "__main__":
    cli()
