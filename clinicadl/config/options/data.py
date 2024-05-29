import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# Data
baseline = click.option(
    "--baseline/--longitudinal",
    default=get_default("baseline", config.DataConfig),
    help="If provided, only the baseline sessions are used for training.",
    show_default=True,
)
diagnoses = click.option(
    "--diagnoses",
    "-d",
    type=get_type("diagnoses", config.DataConfig),
    default=get_default("diagnoses", config.DataConfig),
    multiple=True,
    help="List of diagnoses used for training.",
    show_default=True,
)
multi_cohort = click.option(
    "--multi_cohort/--single_cohort",
    default=get_default("multi_cohort", config.DataConfig),
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
    show_default=True,
)
participants_tsv = click.option(
    "--participants_tsv",
    type=get_type("data_tsv", config.DataConfig),
    default=get_default("data_tsv", config.DataConfig),
    help="Path to a TSV file including a list of participants/sessions.",
    show_default=True,
)
n_subjects = click.option(
    "--n_subjects",
    type=get_type("n_subjects", config.DataConfig),
    default=get_default("n_subjects", config.DataConfig),
    help="Number of subjects in each class of the synthetic dataset.",
)
caps_directory = click.option(
    "--caps_directory",
    type=get_type("caps_directory", config.DataConfig),
    default=get_default("caps_directory", config.DataConfig),
    help="Data using CAPS structure, if different from the one used during network training.",
    show_default=True,
)
label = click.option(
    "--label",
    type=get_type("label", config.DataConfig),
    default=get_default("label", config.DataConfig),
    show_default=True,
    help="Target label used for training (if NETWORK_TASK in [`regression`, `classification`]). "
    "Default will reuse the same label as during the training task.",
)
