from clinicadl.utils.maps_manager import MapsManager

def launch(
    caps_directory,
    maps_directory,
    preprocessing_json,
    use_extracted_features,
    use_gpu,
    n_proc,
    batch_size,
    evaluation_steps,
    multi_cohort,
    diagnoses,
    baseline,
    normalize,
    data_augmentation,
    sampler,
    predict_atlas_intensities,
    atlas_weight,
    merged_tsv,
    n_splits,
    split,
    epochs,
    learning_rate,
    weight_decay,
    dropout,
    patience,
    tolerance,
    accumulation_steps):
    """
    """
    # Read default arguments in config toml file
    # Read input config file if given in cli option
    # Read option manually given in cli
    train_dict = {
        # Model
        "model": class_name,
        "multi": True or False,
        # Computational
        "use_gpu": use_gpu,
        "n_proc": use_gpu,
        "batch_size": use_gpu,
        "evaluation_steps": use_gpu,
        # Data
        "multi_cohort": multi_cohort,
        "diagnoses": diagnoses,
        "baseline": baseline,
        "normalize": normalize,
        "data_augmentation": data_augmentation,
        "sampler": sampler,
        "predict_atlas_intensities": predict_atlas_intensities,
        "atlas_weight": atlas_weight,
        "merged_tsv": merged_tsv,
        # Cross validation
        "n_splits": n_splits,
        "split": split,
        # Optimization
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay":weight_decay,
        "dropout": dropout,
        "patience": patience,
        "tolerance": tolerance,
        "accumulation_steps": tolerance,
    }
    print(train_dict)
    
    maps_manager = MapsManager(maps_directory, train_dict)
    maps_manager.train()
