import click
from pathlib import Path

from clinicadl.utils import cli_param

from clinicadl.abnormality_map.abnormality_map import compute_abnormality_map

@click.command(name="abnormality_map", no_args_is_help=True)
@cli_param.argument.input_maps
@cli_param.argument.data_group
@cli_param.argument.preprocessing_json
@cli_param.option.participant_list
@cli_param.option.selection_metrics
@cli_param.option.split
@cli_param.option.use_gpu
@cli_param.option.n_proc
@cli_param.option.batch_size
@click.option(
    "--multi_cohort",
    type=bool,
    default=False,
    is_flag=True,
    help="""Allow to use multiple CAPS directories.
            In this case, CAPS_DIRECTORY and PARTICIPANTS_TSV must be paths to TSV files.""",
)
@click.option(
    "--abn_map_type", 
    type=str,
    default="residual",
    help="""Type of abnormality map to compute. Default is residual.""",
)
def cli(
    input_maps_directory,
    data_group,
    preprocessing_json,
    participants_tsv,
    split,
    abn_map_type,
    gpu,
    n_proc,
    batch_size,
    selection_metrics,
    multi_cohort,
):
    """Infer the outputs of a trained model on a test set.
    INPUT_MAPS_DIRECTORY is the MAPS folder from where the model used for prediction will be loaded.
    DATA_GROUP is the name of the subjects and sessions list used for the interpretation.
    """
    from clinicadl.utils.cmdline_utils import check_gpu

    if gpu:
        check_gpu()
        
    compute_abnormality_map(
        maps_dir=input_maps_directory,
        abn_map_fn=abn_map_type,
        data_group=data_group,
        tsv_path=participants_tsv,
        gpu=gpu,
        n_proc=n_proc,
        batch_size=batch_size,
        split_list=split,
        selection_metrics=selection_metrics,
        multi_cohort=multi_cohort,
        preprocessing_json=preprocessing_json,
    )