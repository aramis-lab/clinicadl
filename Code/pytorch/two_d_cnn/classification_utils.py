import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os, shutil
from skimage.transform import resize
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def train(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch_i, model_mode="train", global_steps=0):
    """
    This is the function to train, validate or test the model, depending on the model_mode parameter.
    :param model:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch_i:
    :return:
    """
    # main training loop
    acc = 0.0
    loss = 0.0

    subjects = []
    y_ground = []
    y_hat = []
    print("Start %s!" % model_mode)
    if model_mode == "train":
        model.train() ## set the model to training mode
    else:
        model.eval() ## set the model to evaluation mode
        torch.cuda.empty_cache()
        # model.zero_grad()
    print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))
    for i, batch_data in enumerate(data_loader):
        if use_cuda:
            imgs, labels = Variable(batch_data['image'].cuda(), volatile=True), Variable(batch_data['label'].cuda(), volatile=True)
        else:
            imgs, labels = Variable(batch_data['image'], volatile=True), Variable(batch_data['label'], volatile=True)

        ## add the participant_id + session_id
        image_ids = batch_data['image_id']
        subjects.extend(image_ids)

        gound_truth_list = labels.data.cpu().numpy().tolist()
        y_ground.extend(gound_truth_list)

        print('The group true label is %s' % (str(labels)))
        output = model(imgs)

        _, predict = output.topk(1)
        predict_list = predict.data.cpu().numpy().tolist()
        y_hat.extend([item for sublist in predict_list for item in sublist])
        if model_mode == "train" or model_mode == 'valid':
            print("output.device: " + str(output.device))
            print("labels.device: " + str(labels.device))
            print("The predicted label is: " + str(output))
            loss_batch = loss_func(output, labels)
        correct_this_batch = (predict.squeeze(1) == labels).sum().float()
        # To monitor the training process using tensorboard, we only display the training loss and accuracy, the other performance metrics, such
        # as balanced accuracy, will be saved in the tsv file.
        accuracy = float(correct_this_batch) / len(labels)
        acc += accuracy
        loss += loss_batch

        if model_mode == "train":
            print("For batch %d, training loss is : %f" % (i, loss_batch.item()))
            print("For batch %d, training accuracy is : %f" % (i, accuracy))

            writer.add_scalar('classification accuracy', accuracy, i + epoch_i * len(data_loader))
            writer.add_scalar('loss', loss_batch, i + epoch_i * len(data_loader))

        elif model_mode == "valid":
            print("For batch %d, validation accuracy is : %f" % (i, accuracy))
            # print("For batch %d, validation loss is : %f" % (i, loss_batch.item()))

        elif model_mode == "test":
            print("For batch %d, validate accuracy is : %f" % (i, accuracy))
            writer.add_scalar('classification accuracy', accuracy, i + epoch_i * len(data_loader))

        # Unlike tensorflow, in Pytorch, we need to manully zero the graident before each backpropagation step, becase Pytorch accumulates the gradients
        # on subsequent backward passes. The initial designing for this is convenient for training RNNs.
        if model_mode == "train":
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        ## update the global steps
        if model_mode == "train":
            global_steps = i + epoch_i * len(data_loader)

        # delete the temporal varibles taking the GPU memory
        # del imgs, labels
        del imgs, labels, output, predict, gound_truth_list, correct_this_batch, loss_batch
        # Releases all unoccupied cached memory
        torch.cuda.empty_cache()

    accuracy_batch_mean = acc / len(data_loader)
    loss_batch_mean = loss / len(data_loader)

    if model_mode == 'valid':
        writer.add_scalar('classification accuracy', accuracy_batch_mean, global_steps + i)
        writer.add_scalar('loss', loss_batch_mean, global_steps + i)

    del loss_batch_mean
    torch.cuda.empty_cache()

    return subjects, y_ground, y_hat, accuracy_batch_mean, global_steps

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    This is the function to save the best model during validation process
    :param state: the parameters that you wanna save
    :param is_best: if the performance is better than before
    :param checkpoint_dir:
    :param filename:
    :return:
    """
    import shutil, os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(checkpoint_dir, 'model_best.pth.tar'))

def subject_diagnosis_df(subject_session_df):
    """
    Creates a DataFrame with only one occurence of each subject and the most early diagnosis
    Some subjects may not have the baseline diagnosis (ses-M00 doesn't exist)

    :param subject_session_df: (DataFrame) a DataFrame with columns containing 'participant_id', 'session_id', 'diagnosis'
    :return: DataFrame with the same columns as the input
    """
    temp_df = subject_session_df.set_index(['participant_id', 'session_id'])
    subjects_df = pd.DataFrame(columns=subject_session_df.columns)
    for subject, subject_df in temp_df.groupby(level=0):
        session_nb_list = [int(session[5::]) for _, session in subject_df.index.values]
        session_nb_list.sort()
        session_baseline_nb = session_nb_list[0]
        if session_baseline_nb < 10:
            session_baseline = 'ses-M0' + str(session_baseline_nb)
        else:
            session_baseline = 'ses-M' + str(session_baseline_nb)
        row_baseline = list(subject_df.loc[(subject, session_baseline)])
        row_baseline.insert(0, subject)
        row_baseline.insert(1, session_baseline)
        row_baseline = np.array(row_baseline).reshape(1, len(row_baseline))
        row_df = pd.DataFrame(row_baseline, columns=subject_session_df.columns)
        subjects_df = subjects_df.append(row_df)

    subjects_df.reset_index(inplace=True, drop=True)
    return subjects_df


def multiple_time_points(df, subset_df):
    """
    Returns a DataFrame with all the time points of each subject

    :param df: (DataFrame) the reference containing all the time points of all subjects.
    :param subset_df: (DataFrame) the DataFrame containing the subset of subjects.
    :return: mtp_df (DataFrame) a DataFrame with the time points of the subjects of subset_df
    """

    mtp_df = pd.DataFrame(columns=df.columns)
    temp_df = df.set_index('participant_id')
    for idx in subset_df.index.values:
        subject = subset_df.loc[idx, 'participant_id']
        subject_df = temp_df.loc[subject]
        if isinstance(subject_df, pd.Series):
            subject_id = subject_df.name
            row = list(subject_df.values)
            row.insert(0, subject_id)
            subject_df = pd.DataFrame(np.array(row).reshape(1, len(row)), columns=df.columns)
            mtp_df = mtp_df.append(subject_df)
        else:
            mtp_df = mtp_df.append(subject_df.reset_index())

    mtp_df.reset_index(inplace=True, drop=True)
    return mtp_df

def split_subjects_to_tsv(diagnoses_tsv, val_size=0.15, random_state=None):
    """
    Write the tsv files corresponding to the train/val/test splits of all folds

    :param diagnoses_tsv: (str) path to the tsv file with diagnoses
    :param val_size: (float) proportion of the train set being used for validation
    :return: None
    """

    df = pd.read_csv(diagnoses_tsv, sep='\t')
    if 'diagnosis' not in list(df.columns.values):
        raise Exception('Diagnoses file is not in the correct format.')
    # Here we reduce the DataFrame to have only one diagnosis per subject (multiple time points case)
    diagnosis_df = subject_diagnosis_df(df)
    diagnoses_list = list(diagnosis_df.diagnosis)
    unique = list(set(diagnoses_list))
    y = np.array([unique.index(x) for x in diagnoses_list])  # There is one label per diagnosis depending on the order

    sets_dir = path.join(path.dirname(diagnoses_tsv),
                         path.basename(diagnoses_tsv).split('.')[0],
                         'val_size-' + str(val_size))
    if not path.exists(sets_dir):
        os.makedirs(sets_dir)

    # split the train data into training and validation set
    skf_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    indices = next(skf_2.split(np.zeros(len(y)), y))
    train_ind, valid_ind = indices

    df_sub_valid = diagnosis_df.iloc[valid_ind]
    df_sub_train = diagnosis_df.iloc[train_ind]
    df_valid = multiple_time_points(df, df_sub_valid)
    df_train = multiple_time_points(df, df_sub_train)

    df_valid.to_csv(path.join(sets_dir, 'valid.tsv'), sep='\t', index=False)
    df_train.to_csv(path.join(sets_dir, 'train.tsv'), sep='\t', index=False)

def load_split(diagnoses_tsv, val_size=0.15, random_state=None):
    """
    Returns the paths of the TSV files for each set

    :param diagnoses_tsv: (str) path to the tsv file with diagnoses
    :param val_size: (float) the proportion of the training set used for validation
    :return: 3 Strings
        training_tsv
        valid_tsv
    """
    sets_dir = path.join(path.dirname(diagnoses_tsv),
                         path.basename(diagnoses_tsv).split('.')[0],
                         'val_size-' + str(val_size))

    training_tsv = path.join(sets_dir, 'train.tsv')
    valid_tsv = path.join(sets_dir, 'valid.tsv')

    if not path.exists(training_tsv) or not path.exists(valid_tsv):
        split_subjects_to_tsv(diagnoses_tsv, val_size, random_state=random_state)

        training_tsv = path.join(sets_dir, 'train.tsv')
        valid_tsv = path.join(sets_dir, 'valid.tsv')

    return training_tsv, valid_tsv

def check_and_clean(d):

  if os.path.exists(d):
      shutil.rmtree(d)
  os.mkdir(d)

class MRIDataset_slice(Dataset):
    """
    This class reads the CAPS of image processing pipeline of DL

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, tsv, transformations=None, transfer_learning=False, mri_plane=0):
        """
        Args:
            caps_directory (string): the output folder of image processing pipeline.
            tsv (string): the tsv containing three columns, participant_id, session_id and diagnosis.
            transformations (callable, optional): if the data sample should be done some transformations or not, such as resize the image.

        To note, for each view:
            Axial_view = "[:, :, slice_i]"
            Coronal_veiw = "[:, slice_i, :]"
            Saggital_view= "[slice_i, :, :]"

        """
        self.caps_directory = caps_directory
        self.tsv = tsv
        self.transformations = transformations
        self.transfer_learning = transfer_learning
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}
        self.mri_plane = mri_plane

        df = pd.io.parsers.read_csv(tsv, sep='\t')
        if ('diagnosis' != list(df.columns.values)[2]) and ('session_id' != list(df.columns.values)[1]) and (
            'participant_id' != list(df.columns.values)[0]):
            raise Exception('the data file is not in the correct format.')
        participant_list = list(df['participant_id'])
        session_list = list(df['session_id'])
        label_list = list(df['diagnosis'])

        # self.participant_list = participant_list
        # self.session_list = session_list
        # self.label_list = label_list

        ## sagital
        if mri_plane == 0:
            self.slice_participant_list = [ele for ele in participant_list for _ in range(139)]
            self.slice_session_list = [ele for ele in session_list for _ in range(139)]
            self.slice_label_list = [ele for ele in label_list for _ in range(139)]
            self.slices_per_patient = 139

        ## coronal
        elif mri_plane == 1:
            self.slice_participant_list = [ele for ele in participant_list for _ in range(139)]
            self.slice_session_list = [ele for ele in session_list for _ in range(139)]
            self.slice_label_list = [ele for ele in label_list for _ in range(139)]
            self.slices_per_patient = 139

        ## axial
        elif mri_plane == 2:
            self.slice_participant_list = [ele for ele in participant_list for _ in range(139)]
            self.slice_session_list = [ele for ele in session_list for _ in range(139)]
            self.slice_label_list = [ele for ele in label_list for _ in range(139)]
            self.slices_per_patient = 139


    def __len__(self):
        return len(self.slice_participant_list)

    def __getitem__(self, idx):

        img_name = self.slice_participant_list[idx]
        sess_name = self.slice_session_list[idx]
        img_label = self.slice_label_list[idx]
        ## image without intensity normalization
        image_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1', 'preprocessing_dl', img_name + '_' + sess_name + '_space-MNI_res-1x1x1.pt')
        # image with intensity normalization
        # image_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1', 'preprocessing_dl', img_name + '_' + sess_name + '_space-MNI_res-1x1x1_linear_registration.pt')
        # samples = []
        label = self.diagnosis_code[img_label]
        index_slice = idx % self.slices_per_patient
        ### To improve the efficiency, the func extract_slice should be done with pytorch Tensor, not on numpy
        extracted_slice = extract_slice(image_path, index_slice, self.mri_plane, self.transfer_learning)

        # for img in images_list:
        if self.transformations:
            img = self.transformations(extracted_slice)
        sample = {'image_id': img_name + '_' + sess_name, 'image': img, 'label': label}
            # samples.append(sample)

        # if need shuffle the data?
        # random.shuffle(samples)

        return sample


def extract_slice(image_path, index_slice, view, transfer_learning):
    """
    This is a function to grab one slice in each view and create a rgb image for transferring learning: duplicate the slices into R, G, B channel
    :param image_path:
    :param view:
    :param transfer_learning: If False, extract the original slices, otherwise, extract the slices and duplicate 3 times to create a fake RGB image.
    :return:

    To note, for each view:
    Axial_view = "[:, :, slice_i]"
    Coronal_veiw = "[:, slice_i, :]"
    Saggital_view= "[slice_i, :, :]"
    """

    image_tensor = torch.load(image_path)
    ## reshape the tensor, delete the first dimension for slice-level
    image_tensor = image_tensor.view(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3])


    extracted_slices = []
    # slice_list = range(15, image_tensor.shape[view] - 15) # delete the first 20 slice and last 15 slices

    # for i in slice_list:
    ## sagital
    if view == 0:
        slice_select = image_tensor[index_slice, :, :]

    ## coronal
    elif view == 1:
        slice_select = image_tensor[:, index_slice, :]

    ## axial
    elif view == 2:
        slice_select = image_tensor[:, :, index_slice]

    ## convert the slices to images based on if transfer learning or not
    if transfer_learning == False:
        extracted_slice = slice_select.unsqueeze(0) ## shape should be 1 * W * L
    else:
        extracted_slice = torch.stack((slice_select, slice_select, slice_select)) ## shape should be 3 * W * L


    return extracted_slice

class CustomResize(object):
    def __init__(self, trg_size):
        self.trg_size = trg_size

    def __call__(self, img):
        resized_img = self.resize_image(img, self.trg_size)
        return resized_img

    def resize_image(self, img_array, trg_size):
        res = resize(img_array, trg_size, mode='reflect', preserve_range=True, anti_aliasing=False)
        res = res.astype('uint8')
        # type check
        if type(res) != np.ndarray:
            raise "type error!"

        # PIL image cannot handle 3D image, only return ndarray type, which ToTensor accepts
        return res


class CustomToTensor(object):
    def __init__(self):
        pass

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            ## to torch.float32
            img = torch.from_numpy(pic.transpose((2, 0, 1))).float()

            # Pytorch does not work with int type. Here, it just change the visualization, the value itself does not change.
            # return img.float()
            return img

def results_to_tsvs(output_dir, iteration, subject_list, y_truth, y_hat, mode='train'):
    """
    This is a function to trace all subject during training, test and validation, and calculate the performances with different metrics into tsv files.
    :param output_dir:
    :param iteration:
    :param subject_list:
    :param y_truth:
    :param y_hat:
    :return:
    """

    # check if the folder exist
    iteration_dir = os.path.join(output_dir, 'performances', 'iteration-' + str(iteration))
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)
    iteration_subjects_df = pd.DataFrame({'iteration': iteration,
                                                'y': y_truth,
                                                'y_hat': y_hat,
                                                'subject': subject_list})
    iteration_subjects_df.to_csv(os.path.join(iteration_dir, mode + '_subjects.tsv'), index=False, sep='\t', encoding='utf-8')

    results = evaluate_prediction(np.asarray(y_truth), np.asarray(y_hat))
    del results['confusion_matrix']
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(iteration_dir, mode + '_result.tsv'), index=False, sep='\t', encoding='utf-8')

    return iteration_subjects_df, pd.DataFrame(results, index=[0])

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
               'npv': npv,
               'confusion_matrix': {'tp': len(tp), 'tn': len(tn), 'fp': len(fp), 'fn': len(fn)}
               }

    return results