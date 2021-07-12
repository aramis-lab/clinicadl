from clinicadl import MapsManager


def group_backprop(options):
    maps_path = options.model_path
    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(maps_path, verbose=verbose_list[options.verbose])

    if options.caps_dir is None:
        options.caps_dir = maps_manager.caps_directory
    if options.tsv_path is None:
        options.tsv_path = maps_manager.tsv_path
    if options.selection is not None:
        options.selection = [selection[5::] for selection in options.selection]

    maps_manager.interpret(
        caps_directory=options.caps_dir,
        tsv_path=options.tsv_path,
        prefix=options.name,
        selection_metrics=options.selection,
        multi_cohort=options.multi_cohort,
        diagnoses=[options.diagnosis],
        baseline=options.baseline,
        target_node=options.target_node,
        save_individual=False,
        batch_size=options.batch_size,
        num_workers=options.nproc,
        use_cpu=options.use_cpu,
    )
