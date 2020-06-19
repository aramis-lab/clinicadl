# coding: utf8

import abc
import os
import pandas as pd
from clinica.pipelines.machine_learning import base
import clinica.pipelines.machine_learning.voxel_based_io as vbio
import clinica.pipelines.machine_learning.ml_utils as utils
from multiprocessing.pool import ThreadPool
from os import path
import numpy as np
from sklearn.model_selection import StratifiedKFold
import nibabel as nib

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen, Jorge Samper Gonzalez"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


class CAPSInput(base.MLInput):

    def __init__(self, caps_directory, diagnoses_tsv, group_id, image_type, precomputed_kernel=None):
        """
        Args:
            caps_directory:
            subjects_visits_tsv:
            diagnoses_tsv:
            group_id:
            image_type: 'T1', 'fdg', 'av45', 'pib' or 'flute'
            precomputed_kernel:
        """

        self._caps_directory = caps_directory
        self._group_id = group_id
        self._image_type = image_type
        self._images = None
        self._x = None
        self._y = None
        self._kernel = None

        subjects_visits = pd.read_csv(diagnoses_tsv, sep='\t')
        if list(subjects_visits.columns.values) != ['participant_id', 'session_id', 'diagnosis']:
            raise Exception('Subjects and visits file is not in the correct format.')
        self._subjects = list(subjects_visits.participant_id)
        self._sessions = list(subjects_visits.session_id)

        diagnoses = pd.read_csv(diagnoses_tsv, sep='\t')
        if 'diagnosis' not in list(diagnoses.columns.values):
            raise Exception('Diagnoses file is not in the correct format.')
        self._diagnoses = list(diagnoses.diagnosis)

        if image_type not in ['T1', 'fdg', 'av45', 'pib', 'flute', 'dwi']:
            raise Exception("Incorrect image type. It must be one of the values 'T1', 'fdg', 'av45', 'pib', 'flute' or 'dwi'")

        if precomputed_kernel is not None:
            if type(precomputed_kernel) == np.ndarray:
                if precomputed_kernel.shape == (len(self._subjects), len(self._subjects)):
                    self._kernel = precomputed_kernel
                else:
                    raise Exception("""Precomputed kernel provided is not in the correct format.
                    It must be a numpy.ndarray object with number of rows and columns equal to the number of subjects,
                    or a filename to a numpy txt file containing an object with the described format.""")
            elif type(precomputed_kernel == str):
                self._kernel = np.loadtxt(precomputed_kernel)
            else:
                raise Exception("""Precomputed kernel provided is not in the correct format.
                It must be a numpy.ndarray object with number of rows and columns equal to the number of subjects,
                or a filename to a numpy txt file containing an object with the described format.""")

    @abc.abstractmethod
    def get_images(self):
        """
        Returns: a list of filenames
        """
        pass

    @abc.abstractmethod
    def get_x(self):
        """
        Returns: a numpy 2d-array.
        """
        pass

    def get_y(self):
        """
        Returns: a list of integers. Each integer represents a class.
        """
        if self._y is not None:
            return self._y

        unique = list(set(self._diagnoses))
        self._y = np.array([unique.index(x) for x in self._diagnoses])
        return self._y

    def get_kernel(self, kernel_function=utils.gram_matrix_linear, recompute_if_exists=False):
        """
        Returns: a numpy 2d-array.
        """
        if self._kernel is not None and not recompute_if_exists:
            return self._kernel

        if self._x is None:
            self.get_x()

        print("Computing kernel ...")
        self._kernel = kernel_function(self._x)
        print("Kernel computed")
        return self._kernel

    def save_kernel(self, output_dir):
        """
        Args:
            output_dir:
        Returns:
        """
        if self._kernel is not None:
            filename = path.join(output_dir, 'kernel.txt')
            np.savetxt(filename, self._kernel)
            return filename
        raise Exception("Unable to save the kernel. Kernel must have been computed before.")

    @abc.abstractmethod
    def save_weights_as_nifti(self, weights, output_dir):
        pass


class CAPSVoxelBasedInput(CAPSInput):

    def __init__(self, caps_directory, diagnoses_tsv, group_id, image_type, fwhm=0,
                 modulated="on", pvc=None, mask_zeros=True, precomputed_kernel=None):
        """
        Args:
            caps_directory:
            diagnoses_tsv:
            group_id:
            image_type: 'T1', 'fdg', 'av45', 'pib' or 'flute'
            fwhm:
            modulated:
            mask_zeros:
            precomputed_kernel:
        """

        super(CAPSVoxelBasedInput, self).__init__(caps_directory, diagnoses_tsv, group_id,
                                                  image_type, precomputed_kernel=precomputed_kernel)

        self._fwhm = fwhm
        self._modulated = modulated
        self._pvc = pvc
        self._mask_zeros = mask_zeros
        self._orig_shape = None
        self._data_mask = None

        if modulated not in ['on', 'off']:
            raise Exception("Incorrect modulation parameter. It must be one of the values 'on' or 'off'")

    def get_images(self):
        """
        Returns: a list of filenames
        """
        if self._images is not None:
            return self._images

        if self._image_type == 'T1':
            fwhm = '' if self._fwhm == 0 else '_fwhm-%dmm' % int(self._fwhm)

            self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
                                      't1/spm/dartel/group-' + self._group_id,
                                      '%s_%s_T1w_segm-graymatter_space-Ixi549Space_modulated-%s%s_probability.nii.gz'
                                      % (self._subjects[i], self._sessions[i], self._modulated, fwhm))
                            for i in range(len(self._subjects))]
        else:
            pvc = '' if self._pvc is None else '_pvc-%s' % self._pvc
            fwhm = '' if self._fwhm == 0 else '_fwhm-%dmm' % int(self._fwhm)
            suvr = 'pons' if self._image_type == 'fdg' else 'cerebellumPons'

            self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
                                      'pet/preprocessing/group-' + self._group_id,
                                      '%s_%s_task-rest_acq-%s_pet_space-Ixi549Space%s_suvr-%s_mask-brain%s_pet.nii.gz'
                                      % (self._subjects[i], self._sessions[i], self._image_type, pvc, suvr, fwhm))
                            for i in range(len(self._subjects))]

        for image in self._images:
            if not path.exists(image):
                raise Exception("File %s doesn't exists." % image)

        return self._images

    def get_x(self):
        """
        Returns: a numpy 2d-array.
        """
        if self._x is not None:
            return self._x

        print('Loading ' + str(len(self.get_images())) + ' subjects')
        self._x, self._orig_shape, self._data_mask = vbio.load_data(self._images, mask=self._mask_zeros)
        print('Subjects loaded')

        return self._x

    def save_weights_as_nifti(self, weights, output_dir):

        if self._images is None:
            self.get_images()

        output_filename = path.join(output_dir, 'weights.nii.gz')
        data = vbio.revert_mask(weights, self._data_mask, self._orig_shape)
        vbio.weights_to_nifti(data, self._images[0], output_filename)


class KFoldCV(base.MLValidation):

    def __init__(self, ml_algorithm):
        self._ml_algorithm = ml_algorithm
        self._fold_results = []
        self._classifier = None
        self._best_params = None
        self._cv = None

    def validate(self, y, n_folds=10, n_threads=15, splits_indices=None):

        if splits_indices is None:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, )
            self._cv = list(skf.split(np.zeros(len(y)), y))
        else:
            self._cv = splits_indices

        async_pool = ThreadPool(n_threads)
        async_result = {}

        for i in range(n_folds):

            train_index, test_index = self._cv[i]
            async_result[i] = async_pool.apply_async(self._ml_algorithm.evaluate, (train_index, test_index))

        async_pool.close()
        async_pool.join()

        for i in range(n_folds):
            self._fold_results.append(async_result[i].get())

        # save the mean of the best models
        self._classifier, self._best_params = self._ml_algorithm.apply_best_parameters(self._fold_results)

        return self._classifier, self._best_params, self._fold_results

    def save_results(self, output_dir):
        if self._fold_results is None:
            raise Exception("No results to save. Method validate() must be run before save_results().")

        subjects_folds = []
        results_folds = []
        container_dir = path.join(output_dir, 'folds')

        if not path.exists(container_dir):
            os.makedirs(container_dir)

        for i in range(len(self._fold_results)):
            subjects_df = pd.DataFrame({'y': self._fold_results[i]['y'],
                                        'y_hat': self._fold_results[i]['y_hat'],
                                        'y_index': self._fold_results[i]['y_index']})
            subjects_df.to_csv(path.join(container_dir, 'subjects_fold-' + str(i) + '.tsv'),
                               index=False, sep='\t', encoding='utf-8')
            subjects_folds.append(subjects_df)

            results_df = pd.DataFrame({'balanced_accuracy': self._fold_results[i]['evaluation']['balanced_accuracy'],
                                       'auc': self._fold_results[i]['auc'],
                                       'accuracy': self._fold_results[i]['evaluation']['accuracy'],
                                       'sensitivity': self._fold_results[i]['evaluation']['sensitivity'],
                                       'specificity': self._fold_results[i]['evaluation']['specificity'],
                                       'ppv': self._fold_results[i]['evaluation']['ppv'],
                                       'npv': self._fold_results[i]['evaluation']['npv']}, index=['i', ])
            results_df.to_csv(path.join(container_dir, 'results_fold-' + str(i) + '.tsv'),
                              index=False, sep='\t', encoding='utf-8')
            results_folds.append(results_df)

        all_subjects = pd.concat(subjects_folds)
        all_subjects.to_csv(path.join(output_dir, 'subjects.tsv'),
                            index=False, sep='\t', encoding='utf-8')

        all_results = pd.concat(results_folds)
        all_results.to_csv(path.join(output_dir, 'results.tsv'),
                           index=False, sep='\t', encoding='utf-8')

        mean_results = pd.DataFrame(all_results.apply(np.nanmean).to_dict(), columns=all_results.columns, index=[0, ])
        mean_results.to_csv(path.join(output_dir, 'mean_results.tsv'),
                            index=False, sep='\t', encoding='utf-8')
        print("Mean results of the classification:")
        print("Balanced accuracy: %s" % (mean_results['balanced_accuracy'].to_string(index=False)))
        print("specificity: %s" % (mean_results['specificity'].to_string(index=False)))
        print("sensitivity: %s" % (mean_results['sensitivity'].to_string(index=False)))
        print("auc: %s" % (mean_results['auc'].to_string(index=False)))


def extract_indices_from_5_fold(diagnosis_tsv_folder, n_splits, output_dir, baseline=True, diagnoses_list=['AD', 'CN']):
    """
    This is a function to restore the indices of the 5 fold made for CNN based on all the diagnosis tsvs
    :param diagnosis_tsv_folder:
    :param n_splits:
    :param baseline_or_longitudinal:
    :return: the combined tsv file and the indices for 5 fold
    """
    splits_indices = []

    for i in range(n_splits):
        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        train_index = []
        valid_index = []

        train_path = path.join(diagnosis_tsv_folder, 'train_splits-' + str(n_splits),
                               'split-' + str(i))
        valid_path = path.join(diagnosis_tsv_folder, 'validation_splits-' + str(n_splits),
                               'split-' + str(i))
        print("Train", train_path)
        print("Valid", valid_path)

        for diagnosis in diagnoses_list:

            if baseline:
                train_diagnosis_tsv = path.join(train_path, diagnosis + '_baseline.tsv')
            else:
                train_diagnosis_tsv = path.join(train_path, diagnosis + '.tsv')

            valid_diagnosis_tsv = path.join(valid_path, diagnosis + '_baseline.tsv')

            train_diagnosis_df = pd.read_csv(train_diagnosis_tsv, sep='\t')
            valid_diagnosis_df = pd.read_csv(valid_diagnosis_tsv, sep='\t')

            train_df = pd.concat([train_df, train_diagnosis_df])
            valid_df = pd.concat([valid_df, valid_diagnosis_df])

        train_df.reset_index(inplace=True, drop=True)
        valid_df.reset_index(inplace=True, drop=True)

        if i == 0:
            # only concatenate the train + valid for the first fold
            all_df = pd.concat([train_df, valid_df])
            all_df.reset_index(inplace=True, drop=True)

            all_tsv = os.path.join(output_dir, 'all_subjects.tsv')
            all_df.to_csv(all_tsv, index=False, sep='\t', encoding='utf-8')

        # find the index for the training and validation based on the concatenated tsv.
        for j in range(len(train_df)):
            row = train_df.iloc[j]
            for index, row_all in all_df.iterrows():
                if row['participant_id'] == row_all['participant_id'] and row['session_id'] == row_all['session_id'] and row['diagnosis'] == row_all['diagnosis']:
                    train_index.append(index)

        for k in range(len(valid_df)):
            row = valid_df.iloc[k]
            for index, row_all in all_df.iterrows():
                if row['participant_id'] == row_all['participant_id'] and row['session_id'] == row_all['session_id'] and row['diagnosis'] == row_all['diagnosis']:
                    valid_index.append(index)

        # convert the list of index to be an array
        train_index_array = np.asarray(train_index)
        valid_index_array = np.asarray(valid_index)

        # convert the two arrays into a tuple
        index_tuple = (train_index_array, valid_index_array)
        splits_indices.append(index_tuple)

    return splits_indices, all_tsv


def load_data_svm(image_list, mask=True):
    """
    Args:
        image_list:
        mask:
    Returns:
    """
    data = None
    shape = None
    data_mask = None
    first = True

    for i in range(len(image_list)):
        subj = nib.load(image_list[i])
        subj_data = np.nan_to_num(subj.get_data().flatten())

        # Memory allocation for ndarray containing all data to avoid copying the array for each new subject
        if first:
            data = np.ndarray(shape=(len(image_list), subj_data.shape[0]), dtype=float, order='C')
            shape = subj.get_data().shape
            first = False

        data[i, :] = subj_data

    if mask:
        data_mask = (data != 0).sum(axis=0) != 0
        data = data[:, data_mask]

    return data, shape, data_mask


def revert_mask(weights, mask, shape):
    """
    Args:
        weights:
        mask:
        shape:
    Returns:
    """

    z = np.zeros(np.prod(shape))
    z[mask] = weights  # ValueError: NumPy boolean array indexing assignment cannot assign 1636161 input values to the 1634188 output values where the mask is true

    new_weights = np.reshape(z, shape)

    return new_weights


def evaluate_prediction(y, y_hat):

    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0

    tp = []
    tn = []
    fp = []
    fn = []

    for i in range(len(y)):
        if y[i] == 1:
            if y_hat[i] == 1:
                true_positive += 1
                tp.append(i)
            else:
                false_negative += 1
                fn.append(i)
        else:  # -1
            if y_hat[i] == 0:
                true_negative += 1
                tn.append(i)
            else:
                false_positive += 1
                fp.append(i)

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if (true_positive + false_negative) != 0:
        sensitivity = true_positive / (true_positive + false_negative)
    else:
        sensitivity = 0.0

    if (false_positive + true_negative) != 0:
        specificity = true_negative / (false_positive + true_negative)
    else:
        specificity = 0.0

    if (true_positive + false_positive) != 0:
        ppv = true_positive / (true_positive + false_positive)
    else:
        ppv = 0.0

    if (true_negative + false_negative) != 0:
        npv = true_negative / (true_negative + false_negative)
    else:
        npv = 0.0

    balanced_accuracy = (sensitivity + specificity) / 2

    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'sensitivity': sensitivity,
               'specificity': specificity,
               'ppv': ppv,
               'npv': npv
               }

    return results


def save_data(df, output_dir, folder_name):
    """
    Save data so it can be used by the workflow
    :param df:
    :param output_dir:
    :param folder_name:
    :return: path to the tsv files
    """
    import os
    from os import path

    results_dir = path.join(output_dir, 'data', folder_name)
    if not path.exists(results_dir):
        os.makedirs(results_dir)

    df.to_csv(path.join(results_dir, 'all_subjects.tsv'), sep="\t", index=False)

    return path.join(results_dir, 'all_subjects.tsv')


def save_weights(classifier, x, output_dir):

    dual_coefficients = classifier.dual_coef_
    sv_indices = classifier.support_

    weighted_sv = dual_coefficients.transpose() * x[sv_indices]
    weights = np.sum(weighted_sv, 0)

    np.savetxt(path.join(output_dir, 'weights.txt'), weights)

    return weights


def apply_best_parameters_each_split(kernel, x, y, results_list, balanced, n_fold, diagnoses_tsv, output_dir):
    """
    Save the best model for each fold
    :param results_list:
    :return:
    """

    from sklearn.svm import SVC

    best_c = results_list[n_fold]['best_parameter']['c']
    best_bal_acc = results_list[n_fold]['best_parameter']['balanced_accuracy']
    train_index = results_list[n_fold]['x_index']

    if balanced:
        svc = SVC(C=best_c, kernel='precomputed', probability=True, tol=1e-6, class_weight='balanced')
    else:
        svc = SVC(C=best_c, kernel='precomputed', probability=True, tol=1e-6)

    outer_kernel = kernel[train_index, :][:, train_index]
    y_train = y[train_index]

    # save the training data for reconstruction use
    df = pd.read_csv(diagnoses_tsv, sep='\t')
    df_training = df.iloc[train_index]

    result_dir = path.join(output_dir, 'classifier', 'fold_' + str(n_fold))
    if not path.exists(result_dir):
        os.makedirs(result_dir)

    training_tsv = os.path.join(result_dir, 'training_subjects.tsv')
    df_training.to_csv(training_tsv, index=False, sep='\t', encoding='utf-8')

    svc.fit(outer_kernel, y_train)
    # save the weight
    save_weights(svc, x[train_index], result_dir)

    return svc, {'c': best_c, 'balanced_accuracy': best_bal_acc}, train_index
