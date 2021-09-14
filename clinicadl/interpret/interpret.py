from typing import List

from clinicadl import MapsManager


def interpret(
    maps_dir: str,
    data_group: str,
    name: str,
    caps_directory: str,
    tsv_path: str,
    selection_metrics: List[str],
    diagnoses: List[str],
    multi_cohort: bool,
    target_node: int,
    save_individual: bool,
    batch_size: int,
    nproc: int,
    use_cpu: bool,
    verbose=0,
    overwrite: bool = False,
):
    verbose_list = ["warning", "info", "debug"]
    if verbose > 2:
        verbose_str = "debug"
    else:
        verbose_str = verbose_list[verbose]

    maps_manager = MapsManager(maps_dir, verbose=verbose_str)

    maps_manager.interpret(
        data_group=data_group,
        name=name,
        caps_directory=caps_directory,
        tsv_path=tsv_path,
        selection_metrics=selection_metrics,
        diagnoses=diagnoses,
        multi_cohort=multi_cohort,
        target_node=target_node,
        save_individual=save_individual,
        batch_size=batch_size,
        num_workers=nproc,
        use_cpu=use_cpu,
        overwrite=overwrite,
    )
