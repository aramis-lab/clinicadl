from clinicadl import MapsManager


def interpret(model_path,
    caps_directory,
    tsv_path,
    prefix_name,
    selection_metrics,
    multi_cohort,
    diagnosis,
    baseline,
    target_node,
    save_individual,
    batch_size,
    nproc,
    use_cpu,
    verbose=0,
    ):
    maps_path = model_path
    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(maps_path, verbose=verbose_list[verbose])

    maps_manager.interpret(
        caps_directory=caps_directory,
        tsv_path=tsv_path,
        prefix=prefix_name,
        selection_metrics=selection_metrics,
        multi_cohort=multi_cohort,
        diagnoses=[diagnosis],
        baseline=baseline,
        target_node=target_node,
        save_individual=save_individual,
        batch_size=batch_size,
        num_workers=nproc,
        use_cpu=use_cpu,
    )
