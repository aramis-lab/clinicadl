from clinicadl import MapsManager


def interpret(
    maps_dir,
    data_group,
    name,
    caps_directory,
    tsv_path,
    selection_metrics,
    multi_cohort,
    target_node,
    save_individual,
    batch_size,
    nproc,
    use_cpu,
    verbose=0,
):
    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(maps_dir, verbose=verbose_list[verbose])

    maps_manager.interpret(
        data_group=data_group,
        name=name,
        caps_directory=caps_directory,
        tsv_path=tsv_path,
        selection_metrics=selection_metrics,
        multi_cohort=multi_cohort,
        target_node=target_node,
        save_individual=save_individual,
        batch_size=batch_size,
        num_workers=nproc,
        use_cpu=use_cpu,
    )
