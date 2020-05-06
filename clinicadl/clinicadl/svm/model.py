# coding: utf8

from os import path
import numpy as np
import os
from .classification_utils import KFoldCV, CAPSVoxelBasedInput, apply_best_parameters_each_split
from clinica.pipelines.machine_learning import base, algorithm

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen, Jorge Samper Gonzalez"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


class VB_KFold_DualSVM(base.MLWorkflow):

    """
    This is a specific implementation of KFold with DualSVM, in which case the data split has been pre-defined.
    """

    def __init__(self, caps_directory, diagnoses_tsv, group_id, image_type, output_dir, fwhm=8,
                 modulated="on", pvc=None, precomputed_kernel=None, mask_zeros=True, n_threads=15, n_folds=5,
                 grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17), splits_indices=None):

        # Here some parameters selected for this task

        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_folds = n_folds
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range
        self._splits_indices = splits_indices
        self._diagnoses_tsv = diagnoses_tsv
        self._caps_directory = caps_directory
        self._group_id = group_id

        # In this case we are running a voxel based input approach
        #

        self._input = CAPSVoxelBasedInput(caps_directory, diagnoses_tsv, group_id,
                                          image_type, fwhm, modulated, pvc, mask_zeros, precomputed_kernel)

        self._validation = None
        self._algorithm = None

    def run(self):

        # Call on parameters already computed

        x = self._input.get_x()
        y = self._input.get_y()
        kernel = self._input.get_kernel()

        # Now algorithm has been selected, in this case Dual SVM algorithm.
        # Look at algorithm.py to understand the input necessary for each method
        # input parameters were chosen previously

        self._algorithm = algorithm.DualSVMAlgorithm(kernel,
                                                     y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads)
        # Here validation type is selected, it's the K fold cross-validation

        self._validation = KFoldCV(self._algorithm)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads, splits_indices=self._splits_indices, n_folds=self._n_folds)

        # Creation of the path where all the results will be saved

        classifier_dir = path.join(self._output_dir, 'classifier')
        if not path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        # Here we have selected what we want save
        self._algorithm.save_classifier(classifier, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        self._validation.save_results(self._output_dir)
        self._input.save_weights_as_nifti(weights, classifier_dir)
        # save the model for each split only fitting with the training data
        for n_fold in range(self._n_folds):
            svc, _, train_index = apply_best_parameters_each_split(kernel, x, y, results, self._balanced, n_fold, self._diagnoses_tsv, self._output_dir)
            classifier_dir = path.join(self._output_dir, 'classifier', 'fold_' + str(n_fold))
            if not path.exists(classifier_dir):
                os.makedirs(classifier_dir)
            self._algorithm.save_classifier(svc, classifier_dir)

            # save the train index for recontruction purpose
            np.savetxt(path.join(classifier_dir, 'train_index.txt'), train_index)
