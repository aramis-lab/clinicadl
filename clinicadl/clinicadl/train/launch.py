# coding: utf8
from clinicadl import MapsManager


def train(params, erase_existing=True):
    train_dict = vars(params)
    train_dict["minmaxnormalization"] = not params.unnormalize
    train_dict["transfer_path"] = train_dict.pop("transfer_learning_path")
    train_dict["transfer_selection"] = train_dict.pop("transfer_learning_selection")
    if params.n_splits > 1:
        train_dict["validation"] = "KFoldSplit"
    else:
        train_dict["validation"] = "SingleSplit"

    if "func" in train_dict:
        del train_dict["func"]
    maps_dir = train_dict.pop("output_dir")

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

    maps_manager = MapsManager(maps_dir, train_dict, verbose="info")
    maps_manager.train(folds=params.folds, overwrite=erase_existing)
