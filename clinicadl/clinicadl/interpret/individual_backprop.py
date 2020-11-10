from os import path
import os
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt

from clinicadl.tools.deep_learning.iotools import read_json, commandline_to_json, translate_parameters, return_logger
from clinicadl.tools.deep_learning.cnn_utils import get_criterion, sort_predicted
from clinicadl.tools.deep_learning.models import create_model, load_model
from clinicadl.tools.deep_learning.data import load_data_test, return_dataset, get_transforms
from .gradients import VanillaBackProp
"""
Module to test whether an occlusion may prevent a pretrained network from classifying a diagnosis accurately

"""


def individual_backprop(options):

    main_logger = return_logger(options.verbose, "main process")
    options = translate_parameters(options)

    fold_list = [fold for fold in os.listdir(options.model_path) if fold[:5:] == "fold-"]
    if len(fold_list) == 0:
        raise ValueError("No folds were found at path %s" % options.model_path)

    for fold in fold_list:
        main_logger.info(fold)
        for selection in options.selection:
            results_path = path.join(options.model_path, fold, 'gradients',
                                     selection, options.name)

            model_options = argparse.Namespace()
            model_options.gpu = options.gpu
            model_options = read_json(model_options, path.join(options.model_path, 'commandline.json'))
            model_options = translate_parameters(model_options)

            if options.tsv_path is None:
                options.tsv_path = model_options.tsv_path
            if options.input_dir is None:
                options.input_dir = model_options.input_dir
            if options.target_diagnosis is None:
                options.target_diagnosis = options.diagnosis

            criterion = get_criterion(model_options.loss)

            # Data management (remove data not well predicted by the CNN)
            training_df = load_data_test(options.tsv_path, [options.diagnosis], baseline=options.baseline)
            training_df.reset_index(drop=True, inplace=True)

            # Model creation
            _, all_transforms = get_transforms(model_options.mode,
                                               minmaxnormalization=model_options.minmaxnormalization)
            data_example = return_dataset(model_options.mode, options.input_dir,
                                          training_df, model_options.preprocessing,
                                          train_transformations=None, all_transformations=all_transforms,
                                          params=options)

            model = create_model(model_options, data_example.size)
            model_dir = os.path.join(options.model_path, fold, 'outputs', selection)
            model, best_epoch = load_model(model, model_dir, gpu=options.gpu, filename='model_best.pth.tar')
            commandline_to_json(options, logger=main_logger)

            # Keep only subjects who were correctly / wrongly predicted by the network
            training_df = sort_predicted(model, training_df, options.input_dir, model_options,
                                         criterion, options.keep_true,
                                         batch_size=options.batch_size, num_workers=options.num_workers,
                                         gpu=options.gpu)

            # Save the tsv files used for the masking task
            data_path = path.join(results_path, 'data')
            if not path.exists(data_path):
                os.makedirs(data_path)
            training_df.to_csv(path.join(data_path, 'label_train.tsv'), sep='\t', index=False)

            data_train = return_dataset(model_options.mode, options.input_dir,
                                        training_df, model_options.preprocessing,
                                        train_transformations=None, all_transformations=all_transforms,
                                        params=options)

            train_loader = DataLoader(data_train,
                                      batch_size=options.batch_size,
                                      shuffle=True,
                                      num_workers=options.num_workers,
                                      pin_memory=True)

            interpreter = VanillaBackProp(model, gpu=options.gpu)

            for data in train_loader:
                if options.gpu:
                    input_batch = data['image'].cuda()
                else:
                    input_batch = data['image']

                single_path = path.join(results_path, data['participant_id'][0], data['session_id'][0])
                map_np = interpreter.generate_gradients(input_batch,
                                                        data_train.diagnosis_code[options.target_diagnosis])

                if len(data_train.size) == 4:
                    if options.nifti_template_path is not None:
                        image_nii = nib.load(options.nifti_template_path)
                        affine = image_nii.affine
                    else:
                        affine = np.eye(4)

                    map_nii = nib.Nifti1Image(map_np[0, 0, :, :, :], affine)
                    nii_path = path.join(single_path, "outputs", "best_loss")
                    if not path.exists(nii_path):
                        os.makedirs(nii_path)
                    nib.save(map_nii, path.join(nii_path, "occlusion.nii.gz"))
                else:
                    jpg_path = path.join(single_path, "occlusion.jpg")
                    if not path.exists(single_path):
                        os.makedirs(single_path)
                    plt.imshow(map_np[0, 0], cmap="coolwarm", vmin=-options.vmax, vmax=options.vmax)
                    plt.colorbar()
                    plt.savefig(jpg_path)
                    plt.close()
