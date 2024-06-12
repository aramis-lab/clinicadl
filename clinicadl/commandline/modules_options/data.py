import click

from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type

# Data
baseline = click.option(
    "--baseline/--longitudinal",
    default=get_default("baseline", DataConfig),
    help="If provided, only the baseline sessions are used for training.",
    show_default=True,
)
diagnoses = click.option(
    "--diagnoses",
    "-d",
    type=get_type("diagnoses", DataConfig),
    default=get_default("diagnoses", DataConfig),
    multiple=True,
    help="List of diagnoses used for training.",
    show_default=True,
)
multi_cohort = click.option(
    "--multi_cohort/--single_cohort",
    default=get_default("multi_cohort", DataConfig),
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
    show_default=True,
)
participants_tsv = click.option(
    "--participants_tsv",
    type=get_type("data_tsv", DataConfig),
    default=get_default("data_tsv", DataConfig),
    help="Path to a TSV file including a list of participants/sessions.",
    show_default=True,
)
n_subjects = click.option(
    "--n_subjects",
    type=get_type("n_subjects", DataConfig),
    default=get_default("n_subjects", DataConfig),
    help="Number of subjects in each class of the synthetic dataset.",
)
caps_directory = click.option(
    "--caps_directory",
    type=get_type("caps_directory", DataConfig),
    default=get_default("caps_directory", DataConfig),
    help="Data using CAPS structure, if different from the one used during network training.",
    show_default=True,
)
label = click.option(
    "--label",
    type=get_type("label", DataConfig),
    default=get_default("label", DataConfig),
    show_default=True,
    help="Target label used for training (if NETWORK_TASK in [`regression`, `classification`]). "
    "Default will reuse the same label as during the training task.",
)
mask_path = click.option(
    "--mask_path",
    type=get_type("mask_path", DataConfig),
    default=get_default("mask_path", DataConfig),
    help="Path to the extracted masks to generate the two labels. "
    "Default will try to download masks and store them at '~/.cache/clinicadl'.",
    show_default=True,
)
preprocessing_json = click.option(
    "-ej",
    "--preprocessing_json",
    type=get_type("preprocessing_json", DataConfig),
    default=get_default("preprocessing_json", DataConfig),
    help="Name of the JSON file created to describe the tensor extraction. "
    "Default will use format extract_{time_stamp}.json",
)
