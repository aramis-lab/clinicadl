# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018

import collections
import math
import os

import numpy as np
import torch
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset

QC_entry = collections.namedtuple(
    "QC_entry",
    ["id", "status", "qc_files", "variant", "cohort", "subject", "visit", "dist"],
)


def load_full_db(
    qc_db_path,
    data_prefix,
    validate_presence=False,
    feat=3,
    table="qc_all",
    use_variant_dist=None,
):
    """Load complete QC database into memory"""
    import sqlite3

    with sqlite3.connect(qc_db_path) as qc_db:

        query = f"""select q.variant,q.cohort,q.subject,q.visit,q.path,q.xfm,q.pass,d.lin 
            from {table} as q left join xfm_dist as d on q.variant=d.variant and q.cohort=d.cohort and q.subject=d.subject and q.visit=d.visit and q.N=d.N"""

        if use_variant_dist is not None:
            if use_variant_dist:
                query += ' where q.variant = "dist"'
            else:
                query += ' where q.variant != "dist"'

        samples = []
        subjects = []

        for line in qc_db.execute(query):
            variant, cohort, subject, visit, path, xfm, _pass, dist = line

            if _pass == "TRUE":
                status = 1
            else:
                status = 0

            _id = "{}:{}:{}:{}".format(variant, cohort, subject, visit)

            qc_files = []
            for i in range(feat):
                qc_file = "{}/{}/qc/aqc_{}_{}_{}.jpg".format(
                    data_prefix, path, subject, visit, i
                )

                if validate_presence and not os.path.exists(qc_file):
                    print("Check:", qc_file)
                else:
                    qc_files.append(qc_file)
            if dist is None:
                dist = -1  # hack
            if len(qc_files) == feat:
                samples.append(
                    QC_entry(
                        _id,
                        status,
                        qc_files,
                        variant,
                        cohort,
                        subject,
                        visit,
                        float(dist),
                    )
                )

        return samples


def load_qc_images(imgs):
    ret = []
    for i, j in enumerate(imgs):
        try:
            im = io.imread(j)
        except:
            raise NameError(f"Problem reading {j}")
        assert im.shape == (224, 224)
        ret.append(torch.from_numpy(im).unsqueeze_(0).float() / 255.0 - 0.5)
    return ret


def load_minc_images(path, winsorize_low=5, winsorize_high=95):
    import numpy as np
    from minc2_simple import minc2_file

    input_minc = minc2_file(path)
    input_minc.setup_standard_order()

    sz = input_minc.shape

    input_images = [
        input_minc[sz[0] // 2, :, :],
        input_minc[:, :, sz[2] // 2],
        input_minc[:, sz[1] // 2, :],
    ]

    # normalize between 5 and 95th percentile
    _all_voxels = np.concatenate(tuple((np.ravel(i) for i in input_images)))
    # _all_voxels=input_minc[:,:,:] # this is slower
    _min = np.percentile(_all_voxels, winsorize_low)
    _max = np.percentile(_all_voxels, winsorize_high)
    input_images = [(i - _min) * (1.0 / (_max - _min)) - 0.5 for i in input_images]

    # flip, resize and crop
    for i in range(3):
        #
        _scale = min(256.0 / input_images[i].shape[0], 256.0 / input_images[i].shape[1])
        # vertical flip and resize
        input_images[i] = transform.rescale(
            input_images[i][::-1, :],
            _scale,
            mode="constant",
            clip=False,
            anti_aliasing=False,
            multichannel=False,
        )

        sz = input_images[i].shape
        # pad image
        dummy = np.full((256, 256), -0.5)
        dummy[
            int((256 - sz[0]) / 2) : int((256 - sz[0]) / 2) + sz[0],
            int((256 - sz[1]) / 2) : int((256 - sz[1]) / 2) + sz[1],
        ] = input_images[i]

        # crop
        input_images[i] = dummy[16:240, 16:240]

    return [torch.from_numpy(i).float().unsqueeze_(0) for i in input_images]


def load_talairach_mgh_images(path, winsorize_low=5, winsorize_high=95):
    import nibabel as nib
    import numpy as np
    import numpy.linalg as npl
    from nibabel.affines import apply_affine

    # conversion from MNI305 space to ICBM-152 space,according to $FREESURFER_HOME/average/mni152.register.dat
    mni305_to_icb152 = np.array(
        [
            [9.975314e-01, -7.324822e-03, 1.760415e-02, 9.570923e-01],
            [-1.296475e-02, -9.262221e-03, 9.970638e-01, -1.781596e01],
            [-1.459537e-02, -1.000945e00, 2.444772e-03, -1.854964e01],
            [0, 0, 0, 1],
        ]
    )

    icbm152_to_mni305 = npl.inv(mni305_to_icb152)

    img = nib.load(path)
    img_data = img.get_fdata()
    sz = img_data.shape

    # transformation from ICBM152 space to the Voxel space in the Freesurfer MNI305 file
    icbm_to_vox = npl.inv(icbm152_to_mni305 @ img.affine)  #

    icbm_origin = np.array([193 / 2 - 96, 229 / 2 - 132, 193 / 2 - 78])
    icbm_origin_x = icbm_origin + np.array([1, 0, 0])
    icbm_origin_y = icbm_origin + np.array([0, 1, 0])
    icbm_origin_z = icbm_origin + np.array([0, 0, 1])

    center = apply_affine(icbm_to_vox, icbm_origin)

    _x = apply_affine(icbm_to_vox, icbm_origin_x) - center
    _y = apply_affine(icbm_to_vox, icbm_origin_y) - center
    _z = apply_affine(icbm_to_vox, icbm_origin_z) - center

    ix = np.argmax(np.abs(_x))
    iy = np.argmax(np.abs(_y))
    iz = np.argmax(np.abs(_z))

    center_ = np.rint(center).astype(int)

    # transpose according to what we need
    img_data = np.transpose(img_data, axes=[ix, iy, iz])
    center_ = np.take(center_, [ix, iy, iz])
    sz = img_data.shape

    if _x[ix] < 0:
        img_data = np.flip(img_data, axis=0)
        center_[0] = sz[0] - center_[0]
    if _y[iy] < 0:
        img_data = np.flip(img_data, axis=1)
        center_[1] = sz[1] - center_[0]
    if _z[iz] < 0:
        img_data = np.flip(img_data, axis=2)
        center_[2] = sz[2] - center_[0]

    slice_0 = np.take(img_data, center_[0], 0)
    slice_1 = np.take(img_data, center_[1], 1)
    slice_2 = np.take(img_data, center_[2], 2)

    # adjust FOV, need to pad Y by 4 voxels
    # Y-Z
    slice_0 = np.pad(
        slice_0[:, 50 : (50 + 193)],
        ((4, 0), (0, 0)),
        constant_values=(0.0, 0.0),
        mode="constant",
    )[0:229, :]
    # X-Z
    slice_1 = slice_1[31 : (31 + 193), 50 : (50 + 193)]
    # X-Y
    slice_2 = np.pad(
        slice_2[31 : (31 + 193), :],
        ((0, 0), (4, 0)),
        constant_values=(0.0, 0.0),
        mode="constant",
    )[:, 0:229]

    input_images = [slice_2.T, slice_0.T, slice_1.T]

    # normalize between 5 and 95th percentile
    _all_voxels = np.concatenate(tuple((np.ravel(i) for i in input_images)))
    # _all_voxels=input_minc[:,:,:] # this is slower
    _min = np.percentile(_all_voxels, winsorize_low)
    _max = np.percentile(_all_voxels, winsorize_high)
    input_images = [(i - _min) * (1.0 / (_max - _min)) - 0.5 for i in input_images]

    # flip, resize and crop
    for i in range(3):
        #
        _scale = min(256.0 / input_images[i].shape[0], 256.0 / input_images[i].shape[1])
        # vertical flip and resize
        input_images[i] = transform.rescale(
            input_images[i][::-1, :],
            _scale,
            mode="constant",
            clip=False,
            anti_aliasing=False,
            multichannel=False,
        )

        sz = input_images[i].shape
        # pad image
        dummy = np.full((256, 256), -0.5)
        dummy[
            int((256 - sz[0]) / 2) : int((256 - sz[0]) / 2) + sz[0],
            int((256 - sz[1]) / 2) : int((256 - sz[1]) / 2) + sz[1],
        ] = input_images[i]

        # crop
        input_images[i] = dummy[16:240, 16:240]

    return [torch.from_numpy(i).float().unsqueeze_(0) for i in input_images]


def init_cv(dataset, fold=0, folds=8, validation=5, shuffle=False, seed=None):
    """
    Initialize Cross-Validation

    returns three indexes
    """
    n_samples = len(dataset)
    whole_range = np.arange(n_samples)

    if shuffle:
        _state = None
        if seed is not None:
            _state = np.random.get_state()
            np.random.seed(seed)

        np.random.shuffle(whole_range)

        if seed is not None:
            np.random.set_state(_state)

    if folds > 0:
        training_samples = np.concatenate(
            (
                whole_range[0 : math.floor(fold * n_samples / folds)],
                whole_range[math.floor((fold + 1) * n_samples / folds) : n_samples],
            )
        )

        testing_samples = whole_range[
            math.floor(fold * n_samples / folds) : math.floor(
                (fold + 1) * n_samples / folds
            )
        ]
    else:
        training_samples = whole_range
        testing_samples = whole_range[0:0]
    #
    validation_samples = training_samples[0:validation]
    training_samples = training_samples[validation:]

    return (
        [dataset[i] for i in training_samples],
        [dataset[i] for i in validation_samples],
        [dataset[i] for i in testing_samples],
    )


def split_dataset(
    all_samples,
    fold=0,
    folds=8,
    validation=5,
    shuffle=False,
    seed=None,
    sec_samples=None,
):
    """
    Split samples, according to the subject field
    into testing,training and validation subsets
    sec_samples will be used for training subset, if provided
    """
    ### extract subject list
    subjects = set()
    for i in all_samples:
        subjects.add(i.subject)
    if sec_samples is not None:
        for i in sec_samples:
            subjects.add(i.subject)

    # split into three
    training_subjects, validation_subjects, testing_subjects = init_cv(
        sorted(list(subjects)),
        fold=fold,
        folds=folds,
        validation=validation,
        shuffle=shuffle,
        seed=seed,
    )
    training_subjects = set(training_subjects)
    validation_subjects = set(validation_subjects)
    testing_subjects = set(testing_subjects)

    # apply index
    validation = [i for i in all_samples if i.subject in validation_subjects]
    testing = [i for i in all_samples if i.subject in testing_subjects]

    if sec_samples is not None:
        training = [i for i in sec_samples if i.subject in training_subjects]
    else:
        training = [i for i in all_samples if i.subject in training_subjects]

    return training, validation, testing


class QCDataset(Dataset):
    """
    QC images dataset. Uses sqlite3 database to load data
    """

    def __init__(self, dataset, data_prefix, use_ref=False):
        """
        Args:
            root_dir (string): Directory with all the data
            use_ref  (Boolean): use reference images
        """
        super(QCDataset, self).__init__()
        self.use_ref = use_ref
        self.qc_samples = dataset
        self.data_prefix = data_prefix
        #
        self.qc_subjects = set(i.subject for i in self.qc_samples)

        if self.use_ref:
            # TODO: allow specify as parameter?
            self.ref_img = load_qc_images(
                [
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_2.jpg",
                ]
            )

    def __len__(self):
        return len(self.qc_samples)

    def __getitem__(self, idx):
        _s = self.qc_samples[idx]
        # load images
        _images = load_qc_images(_s.qc_files)

        if self.use_ref:
            _images = torch.cat(
                [item for sublist in zip(_images, self.ref_img) for item in sublist]
            )
        else:
            _images = torch.cat(_images)

        return {"image": _images, "status": _s.status, "id": _s.id, "dist": _s.dist}

    def n_subjects(self):
        """
        Return number of unique subjects
        """
        return len(self.qc_subjects)

    def get_balance(self):
        """
        Calculate class balance True/(True+False)
        """
        cnt = np.zeros(2)
        for i in self.qc_samples:
            cnt[i.status] = cnt[i.status] + 1
        return cnt[1] / (cnt[1] + cnt[0]) if (cnt[1] + cnt[0]) > 0 else 0.0

    def balance(self):
        """
        Balance dataset by excluding some samples
        """
        # TODO: shuffle?
        pos_samples = [i for i in self.qc_samples if i.status == 1]
        neg_samples = [i for i in self.qc_samples if i.status == 0]

        n_both = min(len(pos_samples), len(neg_samples))

        self.qc_samples = pos_samples[0:n_both] + neg_samples[0:n_both]
        self.qc_subjects = set(i.subject for i in self.qc_samples)


class MincVolumesDataset(Dataset):
    """
    Minc volumes dataset, loads slices from a list of images
    For inference in batch mode
    Arguments:
        file_list - list of minc files to load
        csv_file - name of csv file to load list from (first column)
    """

    def __init__(
        self,
        file_list=None,
        csv_file=None,
        winsorize_low=5,
        winsorize_high=95,
        use_ref=False,
        data_prefix=None,
    ):
        self.use_ref = use_ref
        self.data_prefix = data_prefix
        self.winsorize_low = winsorize_low
        self.winsorize_high = winsorize_high

        if file_list is not None:
            self.file_list = file_list
        elif csv_file is not None:
            self.file_list = []
            import csv

            for r in csv.reader(open(csv_file, "r")):
                self.file_list.append(r[0])
        else:
            self.file_list = []

        if self.use_ref:
            # TODO: allow specify as parameter?
            self.ref_img = load_qc_images(
                [
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_2.jpg",
                ]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        _images = load_minc_images(
            self.file_list[idx],
            winsorize_low=self.winsorize_low,
            winsorize_high=self.winsorize_high,
        )

        if self.use_ref:
            _images = torch.cat(
                [item for sublist in zip(_images, self.ref_img) for item in sublist]
            )
        else:
            _images = torch.cat(_images)

        return _images.unsqueeze(0), self.file_list[idx]


class QCImagesDataset(Dataset):
    """
    QC images dataset, loads images identified by prefix in csv file
    Used for inference in batch mode only
    Arguments:
        file_list - list of QC images prefixes files to load
        csv_file - name of csv file to load list from (first column should contain prefix )
    """

    def __init__(self, file_list=None, csv_file=None, use_ref=False, data_prefix=None):
        self.use_ref = use_ref
        self.data_prefix = data_prefix

        if file_list is not None:
            self.file_list = file_list
        elif csv_file is not None:
            self.file_list = []
            import csv

            for r in csv.reader(open(csv_file, "r")):
                self.file_list.append(r[0])
        else:
            self.file_list = []

        if self.use_ref:
            # TODO: allow specify as parameter?
            self.ref_img = load_qc_images(
                [
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                    self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_2.jpg",
                ]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        _images = load_qc_images(
            [self.file_list[idx] + "_{}.jpg".format(i) for i in range(3)]
        )

        if self.use_ref:
            _images = torch.cat(
                [item for sublist in zip(_images, self.ref_img) for item in sublist]
            )
        else:
            _images = torch.cat(_images)
        return _images.unsqueeze(0), self.file_list[idx]
