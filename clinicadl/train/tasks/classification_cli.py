import click

from clinicadl.utils.cli_param import train_option

from .task_utils import task_launcher


@click.command(name="classification", no_args_is_help=True)
# Mandatory arguments
@train_option.caps_directory
@train_option.preprocessing_json
@train_option.tsv_directory
@train_option.output_maps
# Options
@train_option.config_file
# Computational
@train_option.gpu
@train_option.n_proc
@train_option.batch_size
@train_option.evaluation_steps
# Reproducibility
@train_option.seed
@train_option.deterministic
@train_option.compensation
# Model
@train_option.architecture
@train_option.multi_network
# Data
@train_option.multi_cohort
@train_option.diagnoses
@train_option.baseline
@train_option.normalize
@train_option.data_augmentation
@train_option.sampler
# Cross validation
@train_option.n_splits
@train_option.split
# Optimization
@train_option.optimizer
@train_option.epochs
@train_option.learning_rate
@train_option.weight_decay
@train_option.dropout
@train_option.patience
@train_option.tolerance
@train_option.accumulation_steps
@train_option.profiler
# transfer learning
@train_option.transfer_path
@train_option.transfer_selection_metric
# Task-related
@train_option.label
@train_option.selection_metrics
@train_option.selection_threshold
@train_option.classification_loss
def cli(**kwargs):
    """
    Train a deep learning model to learn a classification task on neuroimaging data.

    CAPS_DIRECTORY is the CAPS folder from where tensors will be loaded.

    PREPROCESSING_JSON is the name of the JSON file in CAPS_DIRECTORY/tensor_extraction folder where
    all information about extraction are stored in order to read the wanted tensors.

    TSV_DIRECTORY is a folder were TSV files defining train and validation sets are stored.

    OUTPUT_MAPS_DIRECTORY is the path to the MAPS folder where outputs and results will be saved.

    Options for this command can be input by declaring argument on the command line or by providing a
    configuration file in TOML format. For more details, please visit the documentation:
    https://clinicadl.readthedocs.io/en/stable/Train/Introduction/#configuration-file
    """
    task_specific_options = [
        "label",
        "selection_metrics",
        "selection_threshold",
        "loss",
    ]
    task_launcher("classification", task_specific_options, **kwargs)


if __name__ == "__main__":
    cli()
