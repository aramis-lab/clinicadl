import shutil
import torch
import pandas as pd
import numpy as np
import os, math
from os import path
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.autograd import Variable
import torch.nn.functional as F

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
    print("Start for %s!" % model_mode)
    if model_mode == "train":
        model.train() ## set the model to training mode
        print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))
        for i, batch_data in enumerate(data_loader):
            if use_cuda:
                imgs, labels = batch_data['image'].cuda(), batch_data['label'].cuda()
            else:
                imgs, labels = batch_data['image'], batch_data['label']

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

            print("For batch %d, training loss is : %f" % (i, loss_batch.item()))
            print("For batch %d, training accuracy is : %f" % (i, accuracy))

            writer.add_scalar('classification accuracy', accuracy, i + epoch_i * len(data_loader))
            writer.add_scalar('loss', loss_batch, i + epoch_i * len(data_loader))

            # Unlike tensorflow, in Pytorch, we need to manully zero the graident before each backpropagation step, becase Pytorch accumulates the gradients
            # on subsequent backward passes. The initial designing for this is convenient for training RNNs.
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            ## update the global steps
            global_steps = i + epoch_i * len(data_loader)

            # delete the temporal varibles taking the GPU memory
            # del imgs, labels
            del imgs, labels, output, predict, gound_truth_list, correct_this_batch, loss_batch
            # Releases all unoccupied cached memory
            torch.cuda.empty_cache()

        accuracy_batch_mean = acc / len(data_loader)
        loss_batch_mean = loss / len(data_loader)
        del loss_batch_mean
        torch.cuda.empty_cache()

    elif model_mode == "valid":
        model.eval() ## set the model to evaluation mode
        torch.cuda.empty_cache()
        with torch.no_grad():
            ## torch.no_grad() needs to be set, otherwise the accumulation of gradients would explose the GPU memory.
            print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))
            for i, batch_data in enumerate(data_loader):
                if use_cuda:
                    imgs, labels = batch_data['image'].cuda(), batch_data['label'].cuda()
                else:
                    imgs, labels = batch_data['image'], batch_data['label']

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
                print("For batch %d, validation accuracy is : %f" % (i, accuracy))

                # delete the temporal varibles taking the GPU memory
                # del imgs, labels
                del imgs, labels, output, predict, gound_truth_list, correct_this_batch, loss_batch
                # Releases all unoccupied cached memory
                torch.cuda.empty_cache()

            accuracy_batch_mean = acc / len(data_loader)
            loss_batch_mean = loss / len(data_loader)

            writer.add_scalar('classification accuracy', accuracy_batch_mean, global_steps)
            writer.add_scalar('loss', loss_batch_mean, global_steps)

            del loss_batch_mean
            torch.cuda.empty_cache()

    return subjects, y_ground, y_hat, accuracy_batch_mean, global_steps


def train_sparse_ae(autoencoder, data_loader, use_cuda, loss_func, optimizer, writer, epoch_i, options):
    """
    This trains the sparse autoencoder.
    :param autoencoder:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch_i:
    :param global_steps:
    :return:
    """
    print("Start training for sparse autoencoder!")
    # Releases all unoccupied cached memory
    torch.cuda.empty_cache()
    epoch_loss = 0
    sparsity = 0.05
    beta = 3
    print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))
    for i, batch_data in enumerate(data_loader):
        if use_cuda:
            imgs = batch_data['image'].cuda()
        else:
            imgs = batch_data['image']

        ## check if the patch contains no information, which means the patch is at the edge fo the MRI and contains NAN
        if torch.sum(torch.isnan(imgs.view(1, -1))):
            del imgs
            pass

        else:
            decoded, encoded = autoencoder(imgs)
            imgs_flatten = imgs.view(imgs.shape[0], options.patch_size * options.patch_size * options.patch_size)
            loss1 = loss_func(decoded, imgs_flatten) / options.batch_size
            if use_cuda:
                rho = (torch.ones([1, encoded.shape[1]]) * sparsity).cuda()
                rho_hat = torch.sum(encoded, dim=0, keepdim=True).cuda()
            else:
                rho = torch.ones([1, encoded.shape[1]]) * sparsity ## this value should be near to 0.
                rho_hat = torch.sum(encoded, dim=0, keepdim=True)
            ## the sparsity loss
            loss2 = kl_divergence(rho, rho_hat) * beta
            if np.sum(np.isnan(imgs_flatten.detach().numpy())):
                raise Exception('Stop, this is wrong! imgs_flatten')
            if np.sum(np.isnan(decoded.detach().numpy())):
                raise Exception('Stop, this is wrong! decoded')
            if np.sum(np.isnan(rho.detach().numpy())):
                raise Exception('Stop, this is wrong! rho')
            if np.sum(np.isnan(rho_hat.detach().numpy())):
                raise Exception('Stop, this is wrong! rho_hat')
            # kl_div_loss(mean_activitaion, sparsity)
            loss = loss1 + beta * loss2 ## beta indicates the importance of the sparsity loss
            epoch_loss += loss
            print("For batch %d, training loss is : %f" % (i, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## save loss into tensorboardX
            writer.add_scalar('loss', loss, i + epoch_i * len(data_loader))
            ## save memory
            del imgs, decoded, loss, loss1, loss2, encoded, rho, imgs_flatten, rho_hat

    return epoch_loss


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


class MRIDataset_patch(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, caps_directory, data_file, patch_size, stride_size, transformations=None, data_type='from_patch'):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transformations (callable, optional): Optional transformations to be applied on a sample.

        """
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.data_type = data_type

        # Check the format of the tsv file here
        self.df = pd.read_csv(data_file, sep='\t')
        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")
        participant_list = list(self.df['participant_id'])
        session_list = list(self.df['session_id'])
        label_list = list(self.df['diagnosis'])

        ## dynamically calculate the number of patches from each MRI based on the parameters of patch_size & stride_size:
        ## Question posted on: https://discuss.pytorch.org/t/how-to-extract-smaller-image-patches-3d/16837/9
        patch_dims = [math.floor((169 - patch_size) / stride_size + 1), math.floor((208 - patch_size) / stride_size + 1), math.floor((179 - patch_size) / stride_size + 1)]
        self.patchs_per_patient = int(patch_dims[0] * patch_dims[1] * patch_dims[2])
        self.patch_participant_list = [ele for ele in participant_list for _ in range(self.patchs_per_patient)]
        self.patch_session_list = [ele for ele in session_list for _ in range(self.patchs_per_patient)]
        self.patch_label_list = [ele for ele in label_list for _ in range(self.patchs_per_patient)]

    def __len__(self):
        return len(self.patch_participant_list)

    def __getitem__(self, idx):
        img_name = self.patch_participant_list[idx]
        sess_name = self.patch_session_list[idx]
        img_label = self.patch_label_list[idx]
        ## image without intensity normalization
        label = self.diagnosis_code[img_label]
        index_patch = idx % self.patchs_per_patient

        if self.data_type == 'from_MRI':
            image_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1', 'preprocessing_dl', img_name + '_' + sess_name + '_space-MNI_res-1x1x1.pt')
            image = torch.load(image_path)
            ### extract the patch from MRI based on a specific size
            patch = extract_patch_from_mri(image, index_patch, self.patch_size, self.stride_size, self.patchs_per_patient)
        else:
            patch_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1',
                                      'preprocessing_dl',
                                      img_name + '_' + sess_name + '_space-MNI_res-1x1x1_patchsize-' + str(self.patch_size) + '_stride-' + str(self.stride_size) + '_patch-' + str(
                                          index_patch) + '.pt')
            patch = torch.load(patch_path)

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image_id': img_name + '_' + sess_name, 'image': patch, 'label': label}

        return sample

    def session_restriction(self, session):
        """
            Allows to generate a new MRIDataset_patch using some specific sessions only (mostly used for evaluation of test)

            :param session: (str) the session wanted. Must be 'all' or 'ses-MXX'
            :return: (DataFrame) the dataset with the wanted sessions
            """
        from copy import copy

        data_output = copy(self)
        if session == "all":
            return data_output
        else:
            df_session = self.df[self.df.session_id == session]
            df_session.reset_index(drop=True, inplace=True)
            data_output.df = df_session
            if len(data_output) == 0:
                raise Exception("The session %s doesn't exist for any of the subjects in the test data" % session)
            return data_output


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


def extract_patch_from_mri(image_tensor, index_patch, patch_size, stride_size, patchs_per_patient):

    ## use pytorch tensor.upfold to crop the patch.
    patches_tensor = image_tensor.unfold(1, patch_size, stride_size).unfold(2, patch_size, stride_size).unfold(3, patch_size, stride_size).contiguous()
    # the dimension of patch_tensor should be [1, patch_num1, patch_num2, patch_num3, patch_size1, patch_size2, patch_size3]
    patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)
    if patchs_per_patient != patches_tensor.shape[0]:
        raise Exception("Oops, the number of patches were not correctly calculated")

    extracted_patch = patches_tensor[index_patch, ...].unsqueeze_(0) ## add one dimension

    return extracted_patch


def check_and_clean(d):

  if os.path.exists(d):
      shutil.rmtree(d)
  os.mkdir(d)

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

class CustomNormalizeMinMax(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        if isinstance(tensor, torch.Tensor):
            ## normalize to [0, 1]
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

            return tensor
        else:
            raise Exception('CustomNormalizedMinMax needs a torch tensor, but it is not given.')

def kl_divergence(p, q):
    '''
    This is the penalty term quantified by KL divergence.
    ref: http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity
    :param p:
    :param q:
    :return:
    '''
    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))

    return s1 + s2

def extract_slice_img(x):
    """
    This is to extrac a middle slice of the input patch or MRI to check the reconstruction quality
    :param x:
    :return:
    """
    slices = x[:, 0, x.shape[-1] // 2, ...].unsqueeze(1)
    return slices