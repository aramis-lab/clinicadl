from os import path
import os
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import warnings

from clinicadl.tools.deep_learning.iotools import read_json, commandline_to_json, translate_parameters, return_logger
from clinicadl.tools.deep_learning.cnn_utils import get_criterion, sort_predicted
from clinicadl.tools.deep_learning.models import create_model, load_model
from clinicadl.tools.deep_learning.data import load_data_test, return_dataset, get_transforms
from .gradients import VanillaBackProp


def group_backprop(options):

    main_logger = return_logger(options.verbose, "main process")
    options = translate_parameters(options)

    fold_list = [fold for fold in os.listdir(options.model_path) if fold[:5:] == "fold-"]
    if len(fold_list) == 0:
        raise ValueError("No folds were found at path %s" % options.model_path)

    model_options = argparse.Namespace()
    model_options = read_json(model_options, path.join(options.model_path, 'commandline.json'))
    model_options = translate_parameters(model_options)
    model_options.gpu = options.gpu

    if model_options.network_type == "multicnn":
        raise NotImplementedError("The interpretation of multi-CNN is not implemented.")

    if options.tsv_path is None and options.input_dir is None:
        options.multi_cohort = model_options.multi_cohort
    if options.tsv_path is None:
        options.tsv_path = model_options.tsv_path
    if options.input_dir is None:
        options.input_dir = model_options.input_dir
    if options.target_diagnosis is None:
        options.target_diagnosis = options.diagnosis

    for fold in fold_list:
        main_logger.info(fold)
        for selection in options.selection:
            results_path = path.join(options.model_path, fold, 'gradients',
                                     selection, options.name)

            criterion = get_criterion(model_options.loss)

            # Data management (remove data not well predicted by the CNN)
            training_df = load_data_test(options.tsv_path, [options.diagnosis], baseline=options.baseline,
                                         multi_cohort=options.multi_cohort)
            training_df.reset_index(drop=True, inplace=True)

            # Model creation
            _, all_transforms = get_transforms(model_options.mode,
                                               minmaxnormalization=model_options.minmaxnormalization)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_example = return_dataset(model_options.mode, options.input_dir,
                                              training_df, model_options.preprocessing,
                                              train_transformations=None, all_transformations=all_transforms,
                                              prepare_dl=options.prepare_dl, multi_cohort=options.multi_cohort,
                                              params=model_options)

            model = create_model(model_options, data_example.size)
            model_dir = os.path.join(options.model_path, fold, 'models', selection)
            model, best_epoch = load_model(model, model_dir, gpu=options.gpu, filename='model_best.pth.tar')
            options.output_dir = results_path
            commandline_to_json(options, logger=main_logger)

            # Keep only subjects who were correctly / wrongly predicted by the network
            training_df = sort_predicted(model, training_df, options.input_dir, model_options,
                                         criterion, options.keep_true,
                                         batch_size=options.batch_size, num_workers=options.num_workers,
                                         gpu=options.gpu)

            if len(training_df) > 0:

                # Save the tsv files used for the saliency maps
                training_df.to_csv(path.join('data.tsv'), sep='\t', index=False)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data_train = return_dataset(model_options.mode, options.input_dir,
                                                training_df, model_options.preprocessing,
                                                train_transformations=None, all_transformations=all_transforms,
                                                prepare_dl=options.prepare_dl, multi_cohort=options.multi_cohort,
                                                params=model_options)

                train_loader = DataLoader(data_train,
                                          batch_size=options.batch_size,
                                          shuffle=True,
                                          num_workers=options.num_workers,
                                          pin_memory=True)

                interpreter = VanillaBackProp(model, gpu=options.gpu)

                cum_map = 0
                for data in train_loader:
                    if options.gpu:
                        input_batch = data['image'].cuda()
                    else:
                        input_batch = data['image']

                    maps = interpreter.generate_gradients(input_batch,
                                                          data_train.diagnosis_code[options.target_diagnosis])
                    cum_map += maps.sum(axis=0)

                mean_map = cum_map / len(data_train)

                if len(data_train.size) == 4:
                    if options.nifti_template_path is not None:
                        image_nii = nib.load(options.nifti_template_path)
                        affine = image_nii.affine
                    else:
                        affine = np.eye(4)

                    mean_map_nii = nib.Nifti1Image(mean_map[0], affine)
                    nib.save(mean_map_nii, path.join(results_path, "map.nii.gz"))
                else:
                    jpg_path = path.join(results_path, "map.jpg")
                    plt.imshow(mean_map[0], cmap="coolwarm", vmin=-options.vmax, vmax=options.vmax)
                    plt.colorbar()
                    plt.savefig(jpg_path)
                    plt.close()
                np.save(path.join(results_path, "map.npy"), mean_map[0])
            else:
                main_logger.warn("There are no subjects for the given options")
