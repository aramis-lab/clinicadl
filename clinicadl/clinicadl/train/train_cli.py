import click

from clinicadl.utils import cli_param

cmd_name = "train"


@click.command(name=cmd_name)
@cli_param.argument.caps_directory
@cli_param.argument.output_maps
@cli_param.argument.preprocessing_json
## train option
@cli_param.option.use_gpu
@cli_param.option.n_proc
@cli_param.option.batch_size
@click.option(
    "--evaluation_steps", "-esteps",
    type=int, default=0,
    help="Fix the number of iterations to perform before computing an evaluation. Default will only "
    "perform one evaluation at the end of each epoch.",
)
# Mode
@click.option(
    "--use_extracted_features",
    type=bool, default=False, is_flag=True,
    help="""If provided the outputs of extract preprocessing are used, else the whole 
            MRI is loaded.""",
)
#Data
@click.option(
    "--multi_cohort",
    type=bool, default=False, is_flag=True,
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
        
)
@click.option(
    "--diagnoses", "-d",
    type= click.Choice(["AD", "BV", "CN", "MCI", "sMCI", "pMCI"]),
    default=("AD", "CN"),
    multiple=True, show_default=True,
    help="List of diagnoses that will be selected for training.",
)
@click.option(
    "--baseline",
    type=bool, default=False, is_flag=True,
    help="If provided, only the baseline sessions are used for training.",
)
@click.option(
    "--normalize/--unnormalize",
    default=False,
    help="Disable default MinMaxNormalization.",
)
@click.option(
    "--data_augmentation", "-da",
    type=click.Choice(["None", "Noise", "Erasing", "CropPad", "Smoothing"]),
    default=False, multiple=True,
    help="Randomly applies transforms on the training set.",
)
@click.option(
    "--sampler", "-s",
    type=click.Choice(["random", "weighted"]),
    default="random",
    help="Sampler choice (random, or weighted for imbalanced datasets)",
)
@click.option(
    "--predict_atlas_intensities",
    type=click.Choice(["AAL2", "AICHA", "Hammers", "LPBA40", "Neuromorphometrics"]),
    default=None,
    help="Atlases used in t1-volume pipeline to make intensities prediction.",
)
@click.option(
    "--atlas_weight",
    type=float, default=1,
    help="Weight to put on the MSE loss used to compute the error on atlas intensities.",
)
@click.option(
    "--merged_tsv",
    type=click.File(), default="",
    help="Path to the output of clinica iotools merged-tsv (concatenation for multi-cohort). "
        "Can accelerate training if atlas intensities are predicted.",
)
#Cross validation
@click.option(
    "--n_splits",
    type=int, default=0,
    help="If a value is given for k will load data of a k-fold CV. "
        "Default value (0) will load a single split.",
)
@click.option(
    "--split", "-s",
    type=int, default=None,
    multiple=True,
    help="Train the list of given folds. By default train all folds.",
)
#Optimization
@click.option(
    "--epochs",
    type=int, default=20,
    help="Maximum number of epochs.",
)
@click.option(
    "--learning_rate", "-lr",
    type=float, default=1e-4,
    help="Learning rate of the optimization.",
)
@click.option(
    "--weight_decay", "-wd",
    type=float, default=1e-4,
    help="Weight decay value used in optimization.",
)
@click.option(
    "--dropout",
    type=float,
    default=0,
    help="rate of dropout that will be applied to dropout layers in CNN.",
)
@click.option(
    "--patience",
    type=int,
    default=0,
    help="Number of epochs for early stopping patience.",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.0,
    help="Value for the early stopping tolerance.",
)
@click.option(
    "--accumulation_steps", "-asteps",
    type=int, default=1,
    help="Accumulates gradients during the given number of iterations before performing the weight update "
        "in order to virtually increase the size of the batch.",
)
def cli(caps_directory, output_maps, preprocessing_json,
        use_extracted_features,
        use_gpu, n_proc, batch_size, evaluation_steps,
        multi_cohort, diagnoses, baseline, normalize, data_augmentation,
        sampler, predict_atlas_intensities, atlas_weight, merged_tsv,
        n_splits, split,
        epochs, learning_rate, weight_decay, dropout, patience,
        tolerance, accumulation_steps):
    """
    Train a deep learning model on INPUT_CAPS_DIRECTORY data.
    Save the results in OUTPUT_MAPS_DIRECTORY.
    Data will be selected with respect to PREPROCESSING_JSON parameters.
    """
    from .train import launch
    launch(
        caps_directory=caps_directory,
        maps_directory=output_maps,
        preprocessing_json=preprocessing_json,
        use_extracted_features=use_extracted_features,
        use_gpu=use_gpu,
        n_proc=n_proc,
        batch_size=batch_size,
        evaluation_steps=evaluation_steps,
        multi_cohort=multi_cohort,
        diagnoses=diagnoses,
        baseline=baseline,
        normalize=normalize,
        data_augmentation=data_augmentation,
        sampler=sampler,
        predict_atlas_intensities=predict_atlas_intensities,
        atlas_weight=atlas_weight,
        merged_tsv=merged_tsv,
        n_splits=n_splits,
        split=split,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        patience=patience,
        tolerance=tolerance,
        accumulation_steps=accumulation_steps,
    )

if __name__ == "__main__":
    cli()
