# coding: utf8

from os.path import isdir, join, abspath, exists
from os import strerror, makedirs, listdir
import errno
import torch
import pathlib
from clinicadl.tools.deep_learning import create_model, load_model, read_json
from clinicadl.tools.deep_learning.data import return_dataset, get_transforms, compute_num_cnn
from clinicadl.tools.deep_learning.cnn_utils import test, soft_voting_to_tsvs, mode_level_to_tsvs
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader


def classify(caps_dir,
             tsv_path,
             model_path,
             prefix_output='prefix_DB',
             no_labels=False,
             gpu=True,
             prepare_dl=True):
    """
    This function verify the input folders, and the existance of the json file
    then it launch the inference stage from a specific model.

    Parameters:

    params : class clinicadl.tools.dee_learning.iotools.Parameters

    Returns:

    """

    # Verify that paths exist
    caps_dir = abspath(caps_dir)
    model_path = abspath(model_path)
    tsv_path = abspath(tsv_path)

    if not isdir(caps_dir):
        print("Folder containing MRIs was not found, please verify its location.")
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), caps_dir)
    if not isdir(model_path):
        print("A valid model in the path was not found. Donwload them from aramislab.inria.fr")
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), model_path)
    if not exists(tsv_path):
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), tsv_path)

    # Infer json file from model_path (suppose that json file is at the same
    # folder)

    json_file = join(model_path, 'commandline.json')

    if not exists(json_file):
        print("Json file doesn't exist")
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), json_file)

    # Verify if a GPU is available
    if gpu:
        if not torch.cuda.is_available():
            print("GPU classifing is not available in your system, it will use cpu.")
            gpu = False

    inference_from_model(
        caps_dir,
        tsv_path,
        model_path,
        json_file,
        prefix_output,
        no_labels,
        gpu,
        prepare_dl)


def inference_from_model(caps_dir,
                         tsv_path,
                         model_path=None,
                         json_file=None,
                         prefix=None,
                         no_labels=False,
                         gpu=True,
                         prepare_dl=False):
    """
    Inference from previously trained model.

    This functions uses a previously trained model to classify the input(s).
    The model is stored in the variable model_path and it assumes the folder
    structure given by the training stage. Particullary to have a prediction at
    image level, it assumes that results of the validation set are stored in
    the model_path folder in order to perform soft-voiting at the slice/patch
    level and also for multicnn.

    Args:

    caps_dir: folder containing the tensor files (.pt version of MRI)
    tsv_path: file with the name of the MRIs to process (single or multiple)
    model_path: file with the model (pth format).
    json_file: file containing the training parameters.
    output_dir_arg: folder where results are stored. If None it uses current
    structure.
    no_labels: by default is false. In that case, output writes a file named
    measurements.tsv
    gpu: if true, it uses gpu.
    prepare_dl: if true, uses extracted patches/slices otherwise extract them
    on-the-fly.

    Returns:

    Files written in the output folder with prediction results and metrics. By
    default the output folder is named cnn_classification and it is inside the
    model_folder.

    Raises:


    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the trained model folder.")
    options = parser.parse_args([model_path])
    options = read_json(options, json_path=json_file)
    num_cnn = compute_num_cnn(caps_dir, tsv_path, options, "classify")
    print("Load model with these options:")
    print(options)

    # Overwrite options with user input
    options.use_cpu = not gpu
    options.prepare_dl = prepare_dl
    # Define the path
    currentDirectory = pathlib.Path(model_path)
    # Search for 'fold-*' pattern
    currentPattern = "fold-*"

    best_model = {
        'best_acc': 'best_balanced_accuracy',
        'best_loss': 'best_loss'
    }

    # loop depending the number of folds found in the model folder
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        fold_path = join(model_path, fold_dir)
        model_path = join(fold_path, 'models')

        if options.mode_task == 'multicnn':
            for cnn_dir in listdir(model_path):
                if not exists(join(model_path, cnn_dir, best_model['best_acc'], 'model_best.pth.tar')):
                    raise FileNotFoundError(
                        errno.ENOENT,
                        strerror(errno.ENOENT),
                        join(model_path,
                             cnn_dir,
                             best_model['best_acc'],
                             'model_best.pth.tar')
                    )

        else:
            full_model_path = join(model_path, best_model['best_acc'])
            if not exists(join(full_model_path, 'model_best.pth.tar')):
                raise FileNotFoundError(
                    errno.ENOENT,
                    strerror(errno.ENOENT),
                    join(full_model_path, 'model_best.pth.tar'))

        performance_dir = join(fold_path, 'cnn_classification', best_model['best_acc'])
        if not exists(performance_dir):
            makedirs(performance_dir)

        # It launch the corresponding function, depending on the mode.
        infered_classes, metrics = inference_from_model_generic(
            caps_dir,
            tsv_path,
            model_path,
            options,
            num_cnn=num_cnn
        )

        # Prepare outputs
        usr_prefix = str(prefix)

        # Write output files at %mode level
        print("Prediction results and metrics are written in the "
              "following folder: %s" % performance_dir)

        mode_level_to_tsvs(currentDirectory, infered_classes, metrics, fold, best_model['best_acc'], options.mode,
                           dataset=usr_prefix)

        # Soft voting
        if hasattr(options, 'selection_threshold'):
            selection_thresh = options.selection_threshold
        else:
            selection_thresh = 0.8

        # Write files at the image level (for patch, roi and slice).
        # It assumes the existance of validation files to perform soft-voting
        if options.mode in ["patch", "roi", "slice"]:
            soft_voting_to_tsvs(currentDirectory, fold, best_model["best_acc"], options.mode,
                                usr_prefix, num_cnn=num_cnn, selection_threshold=selection_thresh)


def inference_from_model_generic(caps_dir, tsv_path, model_path, model_options,
                                 num_cnn=None, selection="best_balanced_accuracy"):
    '''
    Inference using an image/subject CNN model


    '''
    from os.path import join

    gpu = not model_options.use_cpu

    # Recreate the model with the network described in the json file
    # Initialize the model
    model = create_model(model_options.model,
                         gpu, dropout=model_options.dropout)
    transformations = get_transforms(model_options.mode,
                                     model_options.minmaxnormalization)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if model_options.mode_task == 'multicnn':

        predictions_df = pd.DataFrame()
        metrics_df = pd.DataFrame()

        for n in range(num_cnn):

            dataset = return_dataset(
                model_options.mode,
                caps_dir,
                tsv_path,
                model_options.preprocessing,
                transformations,
                model_options,
                cnn_index=n)

            test_loader = DataLoader(
                dataset,
                batch_size=model_options.batch_size,
                shuffle=False,
                num_workers=model_options.nproc,
                pin_memory=True)

            # load the best trained model during the training
            model, best_epoch = load_model(
                model,
                join(model_path, 'cnn-%i' % n, selection),
                gpu,
                filename='model_best.pth.tar')

            cnn_df, cnn_metrics = test(
                model,
                test_loader,
                gpu,
                criterion,
                mode=model_options.mode)

            predictions_df = pd.concat([predictions_df, cnn_df])
            metrics_df = pd.concat([metrics_df, pd.DataFrame(cnn_metrics, index=[0])])

        predictions_df.reset_index(drop=True, inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)

    else:

        # Load model from path
        best_model, best_epoch = load_model(
            model, join(model_path, selection),
            gpu, filename='model_best.pth.tar')

        # Read/localize the data
        data_to_test = return_dataset(
            model_options.mode,
            caps_dir,
            tsv_path,
            model_options.preprocessing,
            transformations,
            model_options)

        # Load the data
        test_loader = DataLoader(
            data_to_test,
            batch_size=model_options.batch_size,
            shuffle=False,
            num_workers=model_options.nproc,
            pin_memory=True)

        # Run the model on the data
        predictions_df, metrics = test(
            best_model,
            test_loader,
            gpu,
            criterion,
            mode=model_options.mode)

        metrics_df = pd.DataFrame(metrics, index=[0])

    return predictions_df, metrics_df
