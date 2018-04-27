#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1234)
import nibabel as nib
from clinica.pipelines.machine_learning import voxel_based_io as vbio
import os.path as path
from sklearn.model_selection import StratifiedKFold
from clinica.pipelines.machine_learning.input import CAPSVoxelBasedInput ## this is for t1
from clinica.pipelines.machine_learning.input import CAPSInput



class DWIVoxelBasedInput(CAPSInput):
    def __init__(self, caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map, tissue_type, threshold, fwhm=None, mask_zeros=True,
                 precomputed_kernel=None):
        """
        This is a class to grab the outputs from CAPS for DWIVoxel based analysis
        :param caps_directory:
        :param subjects_visits_tsv:
        :param diagnoses_tsv:
        :param dwi_map: should be one of 'fa', 'md', 'rd' and 'ad'
        :param fwhm: the smoothing kernel in mm
        :param tissue_type: should be one of 'GM', 'GM_WM' and 'WM'
        :param threshold: the threshold of the mask
        :param mask_zeros:
        :param precomputed_kernel:
        """

        super(DWIVoxelBasedInput, self).__init__(caps_directory, subjects_visits_tsv, diagnoses_tsv, None,
                                                 image_type='dwi', precomputed_kernel=precomputed_kernel)

        self._mask_zeros = mask_zeros
        self._orig_shape = None
        self._data_mask = None
        self._dwi_map = dwi_map
        self._tissue_type = tissue_type
        self._threshold = threshold
        self._fwhm = fwhm
        # self._balanced_down_sample = balanced_down_sample

        if dwi_map not in ['fa', 'md', 'rd', 'ad']:
            raise Exception("Incorrect DWI map name. It must be one of the values 'fa', 'md', 'rd', 'ad'")

        if tissue_type not in ['GM', 'GM_WM', 'WM']:
            raise Exception("Incorrect tissue type. It must be one of the values 'GM', 'GM_WM', 'WM'")

    def get_images(self):
        """

        Returns: a list of filenames

        """
        if self._images is not None:
            return self._images

        if self._dwi_map == 'fa':
            self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
                                      'dwi/normalized_space', 'SyN_QuickWarped_thresh.nii.gz')
                            for i in range(len(self._subjects))]
        else:
            self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
                                      'dwi/normalized_space', 'space-JHUTracts0_%s_thresh.nii.gz'
                                      % self._dwi_map)
                            for i in range(len(self._subjects))]

        # ### to grab the masked image
        # if self._fwhm == None:
        #     self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
        #                               'dwi/normalized_space', '%s_non_smoothed_%s_masked_threshold-%s.nii.gz'
        #                               % (self._tissue_type, self._dwi_map, str(self._threshold)))
        #                     for i in range(len(self._subjects))]
        # else:
        #     self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
        #                               'dwi/normalized_space', '%s_fwhm-%smm_%s_masked_threshold-%s.nii.gz'
        #                               % (self._tissue_type, str(self._fwhm), self._dwi_map, str(self._threshold)))
        #                     for i in range(len(self._subjects))]


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

        print 'Loading ' + str(len(self.get_images())) + ' subjects'
        self._x, self._orig_shape, self._data_mask = vbio.load_data(self._images, mask=self._mask_zeros)
        print 'Subjects loaded'

        return self._x

    def save_weights_as_nifti(self, weights, output_dir):

        if self._images is None:
            self.get_images()

        output_filename = path.join(output_dir, 'weights.nii.gz')
        data = vbio.revert_mask(weights, self._data_mask, self._orig_shape)

        features = data / abs(data).max()

        img = nib.load(self._images[0])

        output_image = nib.Nifti1Image(features, img.affine)

        nib.save(output_image, output_filename)


def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_dumy_data(fold_index, n_fold):

    np.random.seed(1234)
    A_data = np.random.uniform(-0.5,1.5, (300, 65, 52, 51)).astype('float32')
    B_data = np.random.uniform(0,1, (300, 65, 52, 51)).astype('float32')  # load two classes for classfication

    A_num,sizeX,sizeY,sizeZ = A_data.shape
    B_num,_,_,_ = B_data.shape
    size_input = [1,sizeX,sizeY,sizeZ]
    np.random.seed(1234)
    random_idx = np.random.permutation(A_num+B_num)
    all_data = np.concatenate((A_data,B_data),axis=0)[random_idx]
    labels = np.hstack((np.ones((A_num,)),np.zeros((B_num,))))[random_idx]

    nn =n_fold
    skf = StratifiedKFold(nn)
    train_id = ['']*nn
    test_id = ['']*nn
    a = 0

    for train_index, test_index in skf.split(all_data, labels):
        print("SPLIT iteration:", a+1, "Traing:", train_index, "Test", test_index)
        train_id[a] = train_index
        test_id[a] = test_index
        a = a+1

    testid = test_id[fold_index]
    validid = test_id[fold_index-1]
    trainid = list(set(train_id[fold_index])-set(validid))
    x_train = all_data[trainid]
    y_train = labels[trainid]
    x_test = all_data[testid]
    y_test = labels[testid]
    x_valid = all_data[validid]
    y_valid = labels[validid]

    return x_train, y_train, x_test, y_test, x_valid, y_valid, size_input

def load_adni_mri(fold_index, n_fold, caps_directory, subjects_visits_tsv, diagnoses_tsv, image_type, group_id='ADNIbl',
                 fwhm=0, sizeX=121, sizeY=145, sizeZ=121):

    """
    This is a function to read the caps of spm t1 pipeline, which was included in Clinica machine learning pipeline.

    :param fold_index:
    :param n_fold:
    :param caps_directory:
    :param subjects_visits_tsv:
    :param diagnoses_tsv:
    :param group_id:
    :param image_type:
    :param fwhm:
    :param sizeX:
    :param sizeY:
    :param sizeZ:
    :return:
    """

    if image_type == 'T1':
        caps_reader = CAPSVoxelBasedInput(caps_directory, subjects_visits_tsv, diagnoses_tsv, group_id,
                                                image_type='T1', fwhm=fwhm, modulated="on", pvc=None, mask_zeros=False, precomputed_kernel=None)

    elif image_type == 'dti':
        ### DTI is not included now, I used the function that I created for my paper.
        caps_reader = DWIVoxelBasedInput(caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map='fa', tissue_type='GM_WM', threshold=0.3,
                                         fwhm=None, mask_zeros=False,
                 precomputed_kernel=None)

    x = caps_reader.get_x()
    y = caps_reader.get_y()

    ## reshape the x into [n_subjects, size_width, size_height, size_depth]
    x = np.reshape(x, (x.shape[0], sizeX, sizeY, sizeZ))

    size_input = [1, sizeX, sizeY, sizeZ]

    n_sujets = x.shape[0]
    print "There are %d sujects in this task, the shape of all data is %d * %d * %d * %d" % (n_sujets, n_sujets, x.shape[1], x.shape[2], x.shape[3])
    ## permutate the subjects.
    np.random.seed(1234)
    random_idx = np.random.permutation(n_sujets)

    all_data = x[random_idx]
    labels = y[random_idx]

    ## created a mean image based on all the image
    mean = np.mean(all_data, axis=0)
    sd_image = np.std(all_data, axis=0) # not use the std func

    ## using stratifiedKFold strategy to split data into training, validation and testing dataset
    nn =n_fold
    skf = StratifiedKFold(nn)
    train_id = ['']*nn
    test_id = ['']*nn
    a = 0

    for train_index, test_index in skf.split(all_data, labels):
        print("SPLIT iteration:", a+1, "Traing:", train_index, "Test", test_index)
        train_id[a] = train_index
        test_id[a] = test_index
        a = a+1

    testid = test_id[fold_index]
    validid = test_id[fold_index-1]
    trainid = list(set(train_id[fold_index])-set(validid))
    x_train = all_data[trainid]
    x_train -= mean
    y_train = labels[trainid]
    x_test = all_data[testid]
    x_test -= mean
    y_test = labels[testid]
    x_valid = all_data[validid]
    x_valid -= mean
    y_valid = labels[validid]

    ### The information for training, validation and test dataset.
    print('For fold %d \n' % fold_index)
    print('Training dataset: \n')
    print('In total, we have %d subjects\n' % x_train.shape[0])
    print('In total, we have %d CN\n' % int(y_train.sum()))

    print('test dataset: \n')
    print('In total, we have %d subjects\n' % x_test.shape[0])
    print('In total, we have %d CN\n' % int(y_test.sum()))

    print('Validation dataset: \n')
    print('In total, we have %d subjects\n' % x_valid.shape[0])
    print('In total, we have %d CN\n' % int(y_valid.sum()))

    return x_train, y_train, x_test, y_test, x_valid, y_valid, size_input