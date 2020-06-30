# coding: utf8

from os.path import isdir, join, abspath, exists
from os import strerror, makedirs, listdir
import errno
import torch
import pathlib
from clinicadl.tools.deep_learning import create_model, load_model, read_json
from clinicadl.tools.deep_learning.data import MinMaxNormalization
from clinicadl.tools.deep_learning.cnn_utils import test, soft_voting
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader


def classify(caps_dir,
             tsv_path,
             model_path,
             prefix_output='prefix_DB',
             output_dir=None,
             no_labels=False,
             gpu=True, 
             prepare_dl=True):
    """
    This function reads the command line parameters and point to inference

    Parameters:

    params : class clinicadl.tools.dee_learning.iotools.Parameters

    Returns:

    """

    # Verify that paths exist
    caps_dir = abspath(caps_dir)
    model_path = abspath(model_path)
    print(model_path)
    tsv_path = abspath(tsv_path)

    if not isdir(caps_dir):
        print("Folder containing MRIs was not found, please verify its location")
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), caps_dir)
        print("Folder containing MRIs is not found, please verify its location")
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

    results_df = inference_from_model(
        caps_dir,
        tsv_path,
        model_path,
        json_file,
        prefix_output,
        output_dir,
        no_labels,
        gpu,
        prepare_dl)


def inference_from_model(caps_dir,
                         tsv_path,
                         model_path=None,
                         json_file=None,
                         prefix=None,
                         output_dir=None,
                         no_labels=False,
                         gpu=True,
                         prepare_dl=False):
    """
    Inference from trained model

    This functions uses a previously trained model to classify the input

    Args:

    caps_dir: folder containing the tensor files (.pt version of MRI)
    tsv_path: file with the name of the MRIs to process (single or multiple)
    model_path: file with the model (pth format).
    json_file: file containing the training parameters.
    output_dir: folder where results are stored. If None it uses current 
    structure.
    no_labels: by default is false. In that case, output writes a file named 
    measurements.tsv
    gpu: if true, it uses gpu.
    prepare_dl: if true, uses extracted patches/slices otherwise extract them
    on-the-fly.

    Returns:

    Pandas data frame with a list of subjects and their infered classes
    (predictions).

    Files stored in user-defined or default folders.

    Raises:


    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the trained model folder.")
    options = parser.parse_args([model_path])
    options = read_json(options, "CNN", json_path=json_file)
    print("Load model with these options:")
    print(options)

    # Overwrite options with user input
    options.use_cpu = not gpu
    options.prepare_dl = prepare_dl
    # define the path
    currentDirectory = pathlib.Path(model_path)
    # search for 'fold_*' pattern
    currentPattern = "fold_*"

    best_model = {
            'best_acc': 'best_balanced_accuracy',
            'best_loss': 'best_loss'
            }

    # loop depending the number of folds found in the model folder
    for fold_dir in currentDirectory.glob(currentPattern):
        print(type(str(fold_dir)[-1]))
        split = int(str(fold_dir)[-1])
        model_fold_path = join(model_path, fold_dir)
        full_model_path = join(model_fold_path, 'models', best_model['best_acc'])
        if not exists(join(full_model_path, 'model_best.pth.tar')) and not options.mode == 'patch':
            raise FileNotFoundError(
                errno.ENOENT, strerror(errno.ENOENT), join(full_model_path, 'model_best.pth.tar'))
        
        if options.mode == 'patch':
            for cnn_dir in listdir(full_model_path):
                if not exists(join(full_model_path, cnn_dir, 'model_best.pth.tar')):
                    raise FileNotFoundError(
                            errno.ENOENT, strerror(errno.ENOENT), join(full_model_path, cnn_dir, 'model_best.pth.tar'))


        if output_dir==None:
            output_dir = join(model_fold_path, 'cnn_classification', best_model['best_acc'])
            if not exists(output_dir):
                makedirs(join(model_fold_path, 'cnn_classification', best_model['best_acc']))
        else:
            if not exists(output_dir):
                raise FileNotFoundError(
                    errno.ENOENT, strerror(errno.ENOENT), output_dir)

        print("Results are saved in: %s" % output_dir)
        
        if (options.mode == 'image'):
            infered_classes, metrics = inference_from_image_model(
                caps_dir,
                tsv_path,
                full_model_path,
                options)
        elif (options.mode == 'slice'):
            infered_classes, metrics = inference_from_slice_model(
                caps_dir,
                tsv_path,
                full_model_path,
                options)
        elif (options.mode == 'patch'):
            infered_classes, metrics = inference_from_patch_model(
                caps_dir,
                tsv_path,
                full_model_path,
                options)
        elif (options.mode == 'roi'):
            infered_classes, metrics = inference_from_roi_model()
        else:
            print("Inference for this image mode is not implemented")

        usr_prefix = str(prefix)
        print(infered_classes)
        output_filename = join(output_dir, usr_prefix + '_%s_level_predictions.tsv' % options.mode)
        
        print("%s level balanced accuracy is %f" % (options.mode, metrics['balanced_accuracy']))
        print("Predictions for your image inputs are stored in: %s" % output_filename)
        infered_classes.to_csv(output_filename, index=False, sep='\t')

        
        # Soft voting
        if hasattr(options, 'selection_threshold')
            selection_thresh = options.selection_threshold
        else:
            selection_thresh = 0.8

        
        if options.mode in ["patch", "roi", "slice"]:
            result_tsv = join(output_dir, 'validation_%s_level_result.tsv' % options.mode)
                    
            validation_df = pd.read_csv(result_tsv, sep='\t')

            df_final, metrics = soft_voting(infered_classes, validation_df,
                    options.mode, selection_threshold=selection_thresh)
            df_final.to_csv(join(output_dir, usr_prefix + '_image_level_result.tsv'),
                    index=False, sep='\t')


    return infered_classes


def inference_from_image_model(caps_dir, tsv_path, model_path, model_options):
    '''
    Inference using an image/subject CNN model



    '''
    from clinicadl.tools.deep_learning.data import MRIDataset

    gpu = not model_options.use_cpu
    # Recreate the model with the network described in the json file
    model = create_model(model_options.model, gpu, dropout=model_options.dropout)
    criterion = nn.CrossEntropyLoss()

    # Load model from path
    best_model, best_epoch = load_model(
        model, model_path,
        gpu, filename='model_best.pth.tar')

    if model_options.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    # Read/localize the data
    data_to_test = MRIDataset(
        caps_dir,
        tsv_path,
        preprocessing=model_options.preprocessing,
        transform=transformations)

    # Load the data
    test_loader = DataLoader(
        data_to_test,
        batch_size=model_options.batch_size,
        shuffle=False,
        num_workers=model_options.nproc,
        pin_memory=True)

    # Run the model on the data
    predictions_df, measures = test(
        best_model,
        test_loader,
        gpu,
        criterion,
        model_options.mode)

    return predictions_df, measures


def inference_from_slice_model(caps_dir, tsv_path, model_path, model_options):
    '''
    Inference using a slice CNN model



    '''
    from clinicadl.tools.deep_learning.data import MRIDataset_slice
    import torchvision.transforms as transforms
    # Initialize the model
    print('Do transfer learning with existed model trained on ImageNet.')

    gpu = not model_options.use_cpu

    model = create_model(model_options.model, gpu, dropout=0.8)
    trg_size = (224, 224)  # most of the imagenet pretrained model has this input size

    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H
    # and W are expected to be at least 224. The images have to be loaded in to
    # a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
    # and std = [0.229, 0.224, 0.225].
    transformations = transforms.Compose([MinMaxNormalization(),
                                          transforms.ToPILImage(),
                                          transforms.Resize(trg_size),
                                          transforms.ToTensor()])
    # Define loss and optimizer
    loss = nn.CrossEntropyLoss()

    # Load model from path
    best_model, best_epoch = load_model(
        model, model_path,
        gpu, filename='model_best.pth.tar')

    # Read/localize the data
    data_to_test = MRIDataset_slice(
        caps_dir,
        tsv_path,
        preprocessing=model_options.preprocessing,
        transformations=transformations,
        mri_plane=model_options.mri_plane,
        prepare_dl=model_options.prepare_dl)

    # Load the data
    test_loader = DataLoader(
        data_to_test,
        batch_size=model_options.batch_size,
        shuffle=False,
        num_workers=model_options.nproc,
        pin_memory=True)

    # Run the model on the data
    predictions_df, mesures = test(
        best_model,
        test_loader,
        gpu,
        loss,
        model_options.mode)

    return predictions_df, mesures


def inference_from_patch_model(caps_dir, tsv_path, model_path, model_options):
    '''
    Inference using an image/subject CNN model



    '''
    from clinicadl.tools.deep_learning.data import MRIDataset_patch, MRIDataset_patch_hippocampus
    import torchvision.transforms as transforms
    from os.path import join

    gpu = not model_options.use_cpu

    if model_options.mode_task == 'cnn':
        # Recreate the model with the network described in the json file
        # Initialize the model
        model = create_model(model_options.model, gpu)
        transformations = transforms.Compose([MinMaxNormalization()])

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # Load model from path
        best_model, best_epoch = load_model(
            model, model_path,
            gpu, filename='model_best.pth.tar')

        # Read/localize the data
        if model_options.hippocampus_roi:
            data_to_test = MRIDataset_patch_hippocampus(
                caps_directory,
                tsv_path,
                preprocessing=model_options.preprocessing,
                transformations=transformations)
        else:
            data_to_test = MRIDataset_patch(
                caps_directory,
                tsv_path,
                model_options.patch_size,
                model_options.stride_size,
                transformations=transformations,
                prepare_dl=model_options.prepare_dl,
                patch_index=None,
                preprocessing=model_options.preprocessing)

        # Load the data
        test_loader = DataLoader(
            data_to_test,
            batch_size=model_options.batch_size,
            shuffle=False,
            num_workers=model_options.nproc,
            pin_memory=True)

        # Run the model on the data
        predictions_df, measures = test(
            best_model,
            test_loader,
            gpu,
            criterion,
            full_return=True)

    elif model_options.mode_task == 'multicnn':

        # Recreate the model with the network described in the json file
        # Initialize the model
        model = create_model(model_options.model, gpu)
        transformations = transforms.Compose([MinMaxNormalization()])

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()

        predictions_df = pd.DataFrame()
        metrics = []

        for n in range(model_options.num_cnn):

            dataset = MRIDataset_patch(
                caps_dir,
                tsv_path,
                model_options.patch_size,
                model_options.stride_size,
                transformations=transformations,
                patch_index=n,
                prepare_dl=model_options.prepare_dl,
                preprocessing=model_options.preprocessing)

            test_loader = DataLoader(
                dataset,
                batch_size=model_options.batch_size,
                shuffle=False,
                num_workers=model_options.nproc,
                pin_memory=True)

            # load the best trained model during the training
            model, best_epoch = load_model(
                model,
                join(model_path, 'cnn-%i' % n),
                gpu,
                filename='model_best.pth.tar')

            cnn_df, cnn_metrics = test(model, test_loader, gpu, criterion)
            print("Patch level balanced accuracy is %f" % cnn_metrics['balanced_accuracy'])
            predictions_df =  pd.concat([predictions_df, cnn_df])
            print(type(cnn_metrics))
            metrics.append(cnn_metrics)
        
        predictions_df.reset_index(drop=True, inplace=True)
        # metrics.rest_index(drop=True, inplace=True)

    else:
        print("Mode not defined")

    return predictions_df, metrics
