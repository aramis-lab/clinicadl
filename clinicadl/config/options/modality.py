import click

from clinicadl.config.config.modality import (
    CustomModalityConfig,
    DTIModalityConfig,
    PETModalityConfig,
)
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

tracer = click.option(
    "--tracer",
    default=get_default("tracer", PETModalityConfig),
    type=get_type("tracer", PETModalityConfig),
    help=(
        "Acquisition label if MODALITY is `pet-linear`. "
        "Name of the tracer used for the PET acquisition (trc-<tracer>). "
        "For instance it can be '18FFDG' for fluorodeoxyglucose or '18FAV45' for florbetapir."
    ),
)
suvr_reference_region = click.option(
    "-suvr",
    "--suvr_reference_region",
    default=get_default("suvr_reference_region", PETModalityConfig),
    type=get_type("suvr_reference_region", PETModalityConfig),
    help=(
        "Regions used for normalization if MODALITY is `pet-linear`. "
        "Intensity normalization using the average PET uptake in reference regions resulting in a standardized uptake "
        "value ratio (SUVR) map. It can be cerebellumPons or cerebellumPon2 (used for amyloid tracers) or pons or "
        "pons2 (used for 18F-FDG tracers)."
    ),
)
custom_suffix = click.option(
    "-cn",
    "--custom_suffix",
    default=get_default("custom_suffix", CustomModalityConfig),
    type=get_type("custom_suffix", CustomModalityConfig),
    help=(
        "Suffix of output files if MODALITY is `custom`. "
        "Suffix to append to filenames, for instance "
        "`graymatter_space-Ixi549Space_modulated-off_probability.nii.gz`, or "
        "`segm-whitematter_probability.nii.gz`"
    ),
)
dti_measure = click.option(
    "--dti_measure",
    "-dm",
    type=get_type("dti_measure", DTIModalityConfig),
    help="Possible DTI measures.",
    default=get_default("dti_measure", DTIModalityConfig),
)
dti_space = click.option(
    "--dti_space",
    "-ds",
    type=get_type("dti_space", DTIModalityConfig),
    help="Possible DTI space.",
    default=get_default("dti_space", DTIModalityConfig),
)
