"""Common CLI options used by Clinica pipelines."""

import click
from clinica.utils.pet import LIST_SUVR_REFERENCE_REGIONS

# TSV TOOLS
diagnoses = click.option(
    "--diagnoses",
    "-d",
    # type=click.Choice(
    #     [
    #         "CN",
    #         "sCN",
    #         "usCN",
    #         "pCN",
    #         "ukCN",
    #         "MCI",
    #         "sMCI",
    #         "usMCI",
    #         "ukMCI",
    #         "pMCI",
    #         "rMCI",
    #         "AD",
    #         "sAD",
    #         "usAD",
    #         "ukAD",
    #         "rAD",
    #         "Dementia",
    #     ]
    # ),
    multiple=True,
    default=("AD", "CN"),
    help="Labels selected for the demographic analysis used in the context of Alzheimer's Disease classification.",
)
modality = click.option(
    "--modality",
    "-mod",
    default="t1w",
    type=str,
    help="Modality to select sessions. Sessions which do not include the modality will be excluded.",
)
no_mci_sub_categories = click.option(
    "--no_mci_sub_categories",
    type=bool,
    default=True,
    is_flag=False,
    help="Deactivate default managing of MCI sub-categories to avoid data leakage.",
)
subset_name = click.option(
    "--subset_name",
    type=str,
    show_default=True,
    default="validation",
    help="Name of the subset that is complementary to train.",
)
test_tsv = click.option(
    "--test_tsv",
    "-tt",
    help="Name of the test file in tsv format",
    type=str,
    default=None,
)
caps_directory = click.option(
    "--caps_directory",
    "-c",
    help="input folder of a CAPS compliant dataset",
    type=str,
    default=None,
)
variables_of_interest = click.option(
    "--variables_of_interest",
    "-voi",
    help="Variables of interest that will be kept in the final lists. "
    "Will always keep the group (that correspond to the diagnosis in most case), subgroup (that correspond to the progression of the disease in the case of a progressive disease), age and sex needed for the split procedure.",
    type=str,
    multiple=True,
    default=None,
)
# GENERATE
participant_list = click.option(
    "--participants_tsv",
    type=click.Path(exists=True),
    help="Path to a TSV file including a list of participants/sessions.",
)
n_subjects = click.option(
    "--n_subjects",
    type=int,
    default=300,
    help="Number of subjects in each class of the synthetic dataset.",
)
preprocessing = click.option(
    "--preprocessing",
    type=click.Choice(["t1-linear", "t1-extensive", "pet-linear"]),
    required=True,
    help="Preprocessing used to generate synthetic data.",
)

# Computational
use_gpu = click.option(
    "--gpu/--no-gpu",
    default=True,
    help="Use GPU by default. Please specify `--no-gpu` to force using CPU.",
)
n_proc = click.option(
    "-np",
    "--n_proc",
    type=int,
    default=2,
    show_default=True,
    help="Number of cores used during the task.",
)
batch_size = click.option(
    "--batch_size",
    type=int,
    default=2,
    show_default=True,
    help="Batch size for data loading.",
)

# Extract
save_features = click.option(
    "--save_features",
    type=bool,
    default=False,
    is_flag=True,
    help="""Extract the selected mode to save the tensor. By default, the pipeline only save images and the mode extraction
            is done when images are loaded in the train.""",
)
subjects_sessions_tsv = click.option(
    "-tsv",
    "--subjects_sessions_tsv",
    type=click.Path(exists=True, resolve_path=True),
    help="TSV file containing a list of subjects with their sessions.",
)
extract_json = click.option(
    "-ej",
    "--extract_json",
    type=str,
    default=None,
    help="Name of the JSON file created to describe the tensor extraction. "
    "Default will use format extract_{time_stamp}.json",
)

use_uncropped_image = click.option(
    "-uui",
    "--use_uncropped_image",
    is_flag=True,
    default=False,
    help="Use the uncropped image instead of the cropped image generated by t1-linear or pet-linear.",
)

acq_label = click.option(
    "--acq_label",
    type=click.Choice(["av45", "fdg"]),
    help=(
        "Acquisition label if MODALITY is `pet-linear`. "
        "Name of the label given to the PET acquisition, specifying  the tracer used (acq-<acq_label>). "
        "For instance it can be 'fdg' for fluorodeoxyglucose or 'av45' for florbetapir."
    ),
)
suvr_reference_region = click.option(
    "-suvr",
    "--suvr_reference_region",
    type=click.Choice(LIST_SUVR_REFERENCE_REGIONS),
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
    default="",
    help=(
        "Suffix of output files if MODALITY is `custom`. "
        "Suffix to append to filenames, for instance "
        "`graymatter_space-Ixi549Space_modulated-off_probability.nii.gz`, or "
        "`segm-whitematter_probability.nii.gz`"
    ),
)
# Data group
overwrite = click.option(
    "--overwrite",
    "-o",
    default=False,
    is_flag=True,
    help="Will overwrite data group if existing. Please give caps_directory and partcipants_tsv to"
    " define new data group.",
)
