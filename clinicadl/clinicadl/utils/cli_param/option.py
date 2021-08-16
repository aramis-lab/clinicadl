"""Common CLI options used by Clinica pipelines."""

import click

subjects_sessions_tsv = click.option(
    "-tsv",
    "--subjects_sessions_tsv",
    type=click.Path(exists=True, resolve_path=True),
    help="TSV file containing a list of subjects with their sessions.",
)

# TSV TOOLS
diagnoses = click.option(
    "--diagnoses",
    "-d",
    type=click.Choice(["AD", "CN", "MCI", "sMCI", "pMCI", "BV"]),
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
    type=click.Choice(["t1-linear", "t1-extensive"]),
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
    "--nproc",
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
