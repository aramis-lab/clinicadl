# coding: utf8
from clinicadl import MapsManager


def train_single_cnn(params, erase_existing=True):
    """
    Trains a single CNN and writes:
        - logs obtained with Tensorboard during training,
        - best models obtained according to two metrics on the validation set (loss and balanced accuracy),
        - for patch and roi modes, the initialization state is saved as it is identical across all folds,
        - final performances at the end of the training.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """
    train_dict = vars(params)
    train_dict["caps_directory"] = train_dict.pop("caps_dir")
    train_dict["multi"] = False
    train_dict["selection_metrics"] = ["loss", "BA"]
    train_dict["optimization_metric"] = "CE"
    train_dict["minmaxnormalization"] = not params.unnormalize
    train_dict["transfer_path"] = train_dict.pop("transfer_learning_path")
    train_dict["transfer_selection"] = train_dict.pop("transfer_learning_selection")
    if params.n_splits > 1:
        train_dict["validation"] = "KFoldSplit"
    else:
        train_dict["validation"] = "SingleSplit"

    if "mri_plane" in train_dict:
        train_dict["slice_direction"] = train_dict.pop("mri_plane")

    if "func" in train_dict:
        del train_dict["func"]
    maps_dir = params.output_dir
    del train_dict["output_dir"]

    if "use_extracted_features" in train_dict:
        train_dict["prepare_dl"] = train_dict["use_extracted_features"]
    elif "use_extracted_patches" in train_dict:
        train_dict["prepare_dl"] = train_dict["use_extracted_patches"]
    elif "use_extracted_slices" in train_dict:
        train_dict["prepare_dl"] = train_dict["use_extracted_slices"]
    elif "use_extracted_roi" in train_dict:
        train_dict["prepare_dl"] = train_dict["use_extracted_roi"]
    else:
        train_dict["prepare_dl"] = False

    train_dict["num_workers"] = train_dict.pop("nproc")
    train_dict["optimizer"] = "Adam"
    if "slice_direction" in train_dict:
        train_dict["mri_plane"] = train_dict.pop("slice_direction")

    maps_manager = MapsManager(maps_dir, train_dict, verbose="info")
    maps_manager.train(folds=params.split, overwrite=erase_existing)


def test_single_cnn(
    model,
    output_dir,
    data_loader,
    subset_name,
    split,
    criterion,
    mode,
    logger,
    selection_threshold,
    gpu=False,
):

    for selection in ["best_balanced_accuracy", "best_loss"]:
        # load the best trained model during the training
        model, best_epoch = load_model(
            model,
            os.path.join(output_dir, "fold-%i" % split, "models", selection),
            gpu=gpu,
            filename="model_best.pth.tar",
        )

        results_df, metrics = test(model, data_loader, gpu, criterion, mode)
        logger.info(
            "%s level %s balanced accuracy is %f for model selected on %s"
            % (mode, subset_name, metrics["balanced_accuracy"], selection)
        )

        mode_level_to_tsvs(
            output_dir, results_df, metrics, split, selection, mode, dataset=subset_name
        )

        # Soft voting
        if data_loader.dataset.elem_per_image > 1:
            soft_voting_to_tsvs(
                output_dir,
                split,
                logger=logger,
                selection=selection,
                mode=mode,
                dataset=subset_name,
                selection_threshold=selection_threshold,
            )
        elif mode != "image":
            mode_to_image_tsvs(
                output_dir, split, selection=selection, mode=mode, dataset=subset_name
            )
