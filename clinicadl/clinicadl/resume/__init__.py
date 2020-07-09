def resume(params):
    from clinicadl.tools.deep_learning.iotools import read_json
    import os
    from os import path

    params = read_json(params)

    if not hasattr(params, "tsv_path"):
        raise ValueError("You are trying to resume a job trained with an older version of clinicadl "
                         "that is not supported with resume.")
    
    # Translate parameters
    params.input_dir = params.caps_dir
    params.num_workers = params.nproc
    params.gpu = not params.use_cpu

    fold_list = sorted([int(fold.split("-")[1]) for fold in os.listdir(params.model_path) if fold[:4:] == "fold"])
    # TODO add hidden file to ensure that train ended correctly
    finished_folds = [fold for fold in fold_list if
                      path.exists(path.join(params.model_path, "fold-%i" % fold, "cnn_classification"))]
    stopped_folds = get_stopped_folds(params, fold_list)
    absent_folds = [fold for fold in params.split if fold not in finished_folds and fold not in stopped_folds]

    print("Finished", finished_folds)
    print("Stopped", stopped_folds)
    print("Absent", absent_folds)

    # To ensure retro-compatibility with random search
    params.output_dir = params.model_path
    resume_fn, train_fn = get_resume_train_fn(params.mode_task)

    params.split = stopped_folds
    resume_fn(params)

    params.split = absent_folds
    train_fn(params)


def get_stopped_folds(params, fold_list):
    from os import path
    from clinicadl.tools.deep_learning.data import compute_num_cnn

    if params.mode_task in ["autoencoder", "cnn"]:
        stopped_folds = [fold for fold in fold_list if
                         path.exists(path.join(params.model_path, "fold-%i" % fold, "models", "checkpoint.pth.tar"))]
    elif params.mode_task == "multicnn":
        stopped_folds = list()
        num_cnn = compute_num_cnn(params.input_dir, params.tsv_path, params)
        for fold in fold_list:
            flag = False
            for cnn_index in num_cnn:
                if path.exists(path.join(params.model_path, "fold-%i" % fold,
                                         "models", "cnn-%i" % cnn_index, "checkpoint.pth.tar")) or \
                        not path.exists(path.join(params.model_path, "fold-%i" % fold, "models", "cnn-%i" % cnn_index)):
                    flag = True
            if flag:
                stopped_folds.append(fold)

    else:
        raise ValueError("Cannot find crashed folds with task %s" % params.mode_task)

    return stopped_folds


def get_resume_train_fn(task):
    from .resume_autoencoder import resume_autoencoder
    from .resume_singleCNN import resume_single_cnn
    from clinicadl.train.train_autoencoder import train_autoencoder
    from clinicadl.train.train_singleCNN import train_single_cnn

    if task == "autoencoder":
        return resume_autoencoder, train_autoencoder
    elif task == "cnn":
        return resume_single_cnn, train_single_cnn
    elif task == "multicnn":  # TODO implement resume multi_cnn
        raise NotImplementedError("The resume option for multicnn was not implemented yet.")
    else:
        raise ValueError("Cannot find resuming and training functions for %s" % task)