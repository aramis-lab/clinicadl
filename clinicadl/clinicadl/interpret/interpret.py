from clinicadl import MapsManager


def interpret_cli(options):
    maps_path = options.model_path
    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(maps_path, verbose=verbose_list[options.verbose])

    maps_manager.interpret(
        caps_directory=options.caps_directory,
        tsv_path=options.tsv_path,
        data_group=options.data_group,
        name=options.name,
        selection_metrics=options.selection_metrics,
        multi_cohort=options.multi_cohort,
        target_node=options.target_node,
        save_individual=options.save_individual,
        batch_size=options.batch_size,
        num_workers=options.nproc,
        use_cpu=options.use_cpu,
    )
