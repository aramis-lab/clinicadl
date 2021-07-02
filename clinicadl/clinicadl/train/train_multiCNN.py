# coding: utf8
from clinicadl import MapsManager


def train_multi_cnn(params, erase_existing=True):
    """
    Trains one CNN per patch and writes for each CNN:
        - logs obtained with Tensorboard during training,
        - best models obtained according to two metrics on the validation set (loss and balanced accuracy),
        - final performances at the end of the training.
    Performances are also aggregated at the image level and combines the output of all networks.
    The initialization state is shared across all networks.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """
    train_dict = vars(params)
    train_dict["caps_directory"] = train_dict.pop("caps_dir")
    train_dict["multi"] = True
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


def test_cnn(
    model,
    output_dir,
    data_loader,
    subset_name,
    split,
    criterion,
    cnn_index,
    mode,
    logger,
    gpu=False,
):

    for selection in ["best_balanced_accuracy", "best_loss"]:
        # load the best trained model during the training
        model, best_epoch = load_model(
            model,
            os.path.join(
                output_dir, "fold-%i" % split, "models", "cnn-%i" % cnn_index, selection
            ),
            gpu=gpu,
            filename="model_best.pth.tar",
        )

        results_df, metrics = test(model, data_loader, gpu, criterion, mode)

        logger.info(
            "%s balanced accuracy is %f for %s %i and model selected on %s"
            % (subset_name, metrics["balanced_accuracy"], mode, cnn_index, selection)
        )

        mode_level_to_tsvs(
            output_dir,
            results_df,
            metrics,
            split,
            selection,
            mode,
            dataset=subset_name,
            cnn_index=cnn_index,
        )
