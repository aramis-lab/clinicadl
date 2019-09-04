import torch
import pandas as pd
import numpy as np
import os, math
from torch.utils.data import Dataset
from time import time

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


#################################
# AutoEncoder train / test
#################################


def stacked_ae_learning(model, train_loader, valid_loader, criterion, writer_train, writer_valid, options, fi):
    """
    This aims to train the stacked AEs together for autoencoder
    :param model:
    :param train_loader:
    :param valid_loader:
    :param criterion:
    :param writer_train:
    :param writer_valid:
    :param options:
    :return:
        Return both the pretrained CNN for future use and also the stacked AEs
    """
    from os import path
    from tools.deep_learning.models import AutoEncoder
    from tools.deep_learning import save_checkpoint, load_model
    from copy import deepcopy

    # if the model defined is not already constructed to an AE, then we convert the CNN into an AE
    ae = AutoEncoder(model)

    ae_finetuning(ae, train_loader, valid_loader, criterion, writer_train, writer_valid, options, fi)

    # Updating and setting weights of the convolutional layers
    checkpoint_dir = path.join(options.output_dir, 'best_model_dir', "fold_" + str(fi), 'ConvAutoencoder',
                               'AutoEncoder', 'best_loss')
    best_autodecoder, best_epoch = load_model(ae, checkpoint_dir, options.gpu,  filename='model_best.pth.tar')

    del ae

    # save the encoder part of the AEs, the best AEs has been saved in the ae_finetuning part
    model.features = deepcopy(best_autodecoder.encoder)
    save_checkpoint({'model': model.state_dict(),
                     'epoch': best_epoch},
                    False, False,
                    os.path.join(options.output_dir, 'best_model_dir', "fold_" + str(fi), 'ConvAutoencoder', 'Encoder'),
                    filename='model_best_encoder.pth.tar')

    del best_epoch

    return model, best_autodecoder


def ae_finetuning(auto_encoder_all, train_loader, valid_loader, criterion, writer_train_ft, writer_valid_ft, options, fi, global_step=0):
    """
    After training the AEs in a layer-wise way, we fine-tune the whole AEs
    :param auto_encoder:
    :param train_loader:
    :param valid_loader:
    :param criterion:
    :param gpu:
    :param results_path:
    :param options:
    :return:
    """
    from tools.deep_learning import save_checkpoint

    auto_encoder_all.train()
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, auto_encoder_all.parameters()),
                                                         options.learning_rate)
    if options.gpu:
        auto_encoder_all.cuda()

    # Initialize variables
    best_loss_valid = np.inf
    print("Beginning fine-tuning")

    print('The number of batches in this sampler based on the batch size: %s' % str(len(train_loader)))
    tend = time()
    total_time = 0

    for epoch in range(options.epochs):
        print("Fine-tuning at %d-th epoch." % epoch)

        auto_encoder_all.zero_grad()

        for i, data in enumerate(train_loader):
            t0 = time()
            total_time = total_time + t0 - tend

            if options.gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']

            train_output = auto_encoder_all(imgs)

            loss = criterion(train_output, imgs)
            loss.backward()

            # monitor the training loss for each batch using tensorboardX
            writer_train_ft.add_scalar('loss', loss, i + epoch * len(train_loader))

            # update the global steps
            global_step = i + epoch * len(train_loader)

            del imgs, train_output, loss

            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

            tend = time()

        print('Mean time per batch (train):', total_time / len(train_loader))

        # Always test the results and save them once at the end of the epoch
        loss_valid = test_ae(auto_encoder_all, valid_loader, options, criterion)
        mean_loss_valid = loss_valid / (len(valid_loader))
        writer_valid_ft.add_scalar('loss', mean_loss_valid, global_step)
        print("Mean validation loss is %f for the -th batch %d" % (mean_loss_valid, global_step))

        # reset the model to train mode after evaluation
        auto_encoder_all.train()

        is_best_loss = loss_valid < best_loss_valid
        # Save best based on smallest loss
        best_loss_valid = min(loss_valid, best_loss_valid)
        save_checkpoint({'model': auto_encoder_all.state_dict(),
                         'iteration': i,
                         'epoch': epoch,
                         'best_loss': best_loss_valid},
                        False, is_best_loss,
                        os.path.join(options.output_dir, "best_model_dir", "fold_" + str(fi), "ConvAutoencoder",
                                     "AutoEncoder"))

    del optimizer, auto_encoder_all


def test_ae(model, dataloader, options, criterion, former_layer=None):
    """
    Computes the loss of the model, either the loss of the layer-wise AE or all the AEs in a big graph one time.

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param gpu: if True a gpu is used
    :return: loss of the model (float)
    """
    model.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if options.gpu:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        if former_layer is not None:
            hidden = former_layer(inputs)
        else:
            hidden = inputs
        outputs = model(hidden)
        ## explicitly set the variable of criterion to be requires_grad=False
        hidden_requires_grad_no = hidden.detach()
        hidden_requires_grad_no.requires_grad = False
        loss = criterion(outputs, hidden_requires_grad_no)
        total_loss += loss.item()
        torch.cuda.empty_cache()

        del inputs, outputs, loss

    return total_loss


#################################
# Transfer learning
#################################

def load_model_after_ae(model, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    This is to copy the weight of the pretrained AE to the current CNN
    :param model:
    :param checkpoint_dir:
    :param filename:
    :return:
    """
    from copy import deepcopy

    model_after_ae = deepcopy(model)
    model_dict = model_after_ae.state_dict()
    param_dict = torch.load(os.path.join(checkpoint_dir, filename))
    ae_pretrained_dict = param_dict['model']
    ae_pretrained_dict_copy = deepcopy(ae_pretrained_dict)

    # remove the classifier's weight, only take the AE
    for k in ae_pretrained_dict.keys():
        if 'classifier' not in k:
            pass
        else:
            del ae_pretrained_dict_copy[k]

    model_dict.update(ae_pretrained_dict_copy)
    model_after_ae.load_state_dict(model_dict)

    return model_after_ae, param_dict['epoch']


def load_model_after_cnn(model, checkpoint_dir, filename='checkpoint.pth.tar'):
    """

    :param model:
    :param checkpoint_dir:
    :param filename:
    :return:
    """
    from copy import deepcopy

    model.eval()
    model_updated = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, filename))
    model_updated.load_state_dict(param_dict['model'])

    return model_updated, param_dict['epoch']


#################################
# CNN train / test
#################################

def train(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch_i, model_mode="train"):
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

    print("Start for %s!" % model_mode)
    global_step = None
    if model_mode == "train":
        columns = ['participant_id', 'session_id', 'patch_index', 'true_label', 'predicted_label', 'proba0', 'proba1']
        results_df = pd.DataFrame(columns=columns)
        total_loss = 0.0

        model.train()  # set the model to training mode
        print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))

        for i, data in enumerate(data_loader):
            # update the global steps
            global_step = i + epoch_i * len(data_loader)

            if use_cuda:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            gound_truth_list = labels.data.cpu().numpy().tolist()

            output = model(imgs)
            _, predicted = torch.max(output.data, 1)
            predict_list = predicted.data.cpu().numpy().tolist()
            batch_loss = loss_func(output, labels)
            total_loss += batch_loss.item()

            # calculate the batch balanced accuracy and loss
            batch_metrics = evaluate_prediction(gound_truth_list, predict_list)
            batch_accuracy = batch_metrics['balanced_accuracy']

            writer.add_scalar('classification accuracy', batch_accuracy, global_step)
            writer.add_scalar('loss', batch_loss.item(), global_step)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx], data['patch_id'][idx],
                       labels[idx].item(), predicted[idx].item(),
                       output[idx, 0].item(), output[idx, 1]]
                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            # delete the temporary variables taking the GPU memory
            del imgs, labels, output, predicted, batch_loss, batch_accuracy
            torch.cuda.empty_cache()

        epoch_metrics = evaluate_prediction(results_df.true_label.values.astype(int),
                                            results_df.predicted_label.values.astype(int))
        accuracy_batch_mean = epoch_metrics['balanced_accuracy']
        loss_batch_mean = total_loss / len(data_loader)
        torch.cuda.empty_cache()

    elif model_mode == "valid":
        results_df, metrics_batch = test(model, data_loader, use_cuda, loss_func)

        # calculate the balanced accuracy
        _, metrics_subject = soft_voting(results_df, results_df)
        accuracy_batch_mean = metrics_subject['balanced_accuracy']
        total_loss = metrics_batch['total_loss']
        loss_batch_mean = total_loss / len(data_loader)

        writer.add_scalar('classification accuracy', accuracy_batch_mean, epoch_i)
        writer.add_scalar('loss', loss_batch_mean, epoch_i)

        torch.cuda.empty_cache()

    else:
        raise ValueError('This mode %s was not implemented. Please choose between train and valid' % model_mode)

    return results_df, accuracy_batch_mean, loss_batch_mean, global_step


def test(model, data_loader, use_cuda, loss_func):
    """
    The function to evaluate the testing data for the trained classifiers
    :param model:
    :param data_loader:
    :param use_cuda:
    :return:
    """

    columns = ['participant_id', 'session_id', 'patch_id', 'true_label', 'predicted_label', 'proba0', 'proba1']
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    print("Start evaluate the model!")
    if use_cuda:
        model.cuda()

    model.eval()  # set the model to evaluation mode
    torch.cuda.empty_cache()
    with torch.no_grad():
        print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))
        for i, data in enumerate(data_loader):
            if use_cuda:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            output = model(imgs)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx], data['patch_id'][idx].item(),
                       labels[idx].item(), predicted[idx].item(),
                       output[idx, 0].item(), output[idx, 1].item()]

                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            del imgs, labels, output
            torch.cuda.empty_cache()

        # calculate the balanced accuracy
        results = evaluate_prediction(results_df.true_label.values.astype(int),
                                      results_df.predicted_label.values.astype(int))
        results['total_loss'] = total_loss
        torch.cuda.empty_cache()

    return results_df, results


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

#################################
# Voting systems
#################################


def patch_level_to_tsvs(output_dir, results_df, results, fold, selection, dataset='train', cnn_index=None):
    """
    Allows to save the outputs of the test function.

    :param output_dir: (str) path to the output directory.
    :param results_df: (DataFrame) the individual results per patch.
    :param results: (dict) the performances obtained on a series of metrics.
    :param fold: (int) the fold for which the performances were obtained.
    :param selection: (str) the metrics on which the model was selected (best_acc, best_loss)
    :param dataset: (str) the dataset on which the evaluation was performed.
    :param cnn_index: (int) provide the cnn_index only for a multi-cnn framework.
    :return:
    """
    if cnn_index is None:
        performance_dir = os.path.join(output_dir, 'performances', 'fold_' + str(fold), selection)
    else:
        performance_dir = os.path.join(output_dir, 'performances', 'fold_' + str(fold), 'cnn-' + str(cnn_index),
                                       selection)

    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)

    results_df.to_csv(os.path.join(performance_dir, dataset + '_patch_level_result-patch_index.tsv'), index=False,
                      sep='\t')

    del results['confusion_matrix']
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(performance_dir, dataset + '_patch_level_metrics.tsv'),
                                            index=False, sep='\t')


def retrieve_patch_level_results(output_dir, fold, selection, dataset, num_cnn):
    """Retrieve performance_df for single or multi-CNN framework."""
    if num_cnn is None:
        result_tsv = os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection,
                                  dataset + '_patch_level_result-patch_index.tsv')
        performance_df = pd.read_csv(result_tsv, sep='\t')

    else:
        performance_df = pd.DataFrame()
        for cnn in range(num_cnn):
            tsv_path = os.path.join(output_dir, 'performances', 'fold_%i' % fold, 'cnn-%i' % cnn, selection,
                                    dataset + '_patch_level_result-patch_index.tsv')
            cnn_df = pd.read_csv(tsv_path, sep='\t')
            performance_df = pd.concat([performance_df, cnn_df])

    return performance_df


def soft_voting_to_tsvs(output_dir, fold, selection, dataset='test', num_cnn=None):
    """
    This is for soft voting for subject-level performances
    :param performance_df: the pandas dataframe, including columns: iteration, y, y_hat, subject, probability
    :param selection: (str) the metrics on which the model was selected (best_acc, best_loss)

    ref: S. Raschka. Python Machine Learning., 2015
    :return:
    """

    # Choose which dataset is used to compute the weights of soft voting.
    if dataset in ['train', 'validation']:
        validation_dataset = dataset
    else:
        validation_dataset = 'validation'
    test_df = retrieve_patch_level_results(output_dir, fold, selection, dataset, num_cnn)
    validation_df = retrieve_patch_level_results(output_dir, fold, selection, validation_dataset, num_cnn)

    df_final, metrics = soft_voting(test_df, validation_df)

    df_final.to_csv(os.path.join(os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection,
                                              dataset + '_subject_level_result_soft_vote.tsv')), index=False, sep='\t')

    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection,
                                                         dataset + '_subject_level_metrics_soft_vote.tsv'),
                                            index=False, sep='\t')


def soft_voting(performance_df, validation_df):

    # selected the right classified subjects on the validation set:
    right_classified_df = validation_df[validation_df['true_label'] == validation_df['predicted_label']]

    # count the number of right classified patch for each patch index
    count_patchs_series = right_classified_df['patch_id'].value_counts(normalize=True)
    index_series = performance_df['patch_id']
    weight_list = []
    for i in index_series:
        if i in count_patchs_series.index:
            weight = count_patchs_series[i]
        else:
            weight = 0
        weight_list.append(weight)

    weight_series = pd.Series(weight_list)
    # add to the df
    performance_df['weight'] = weight_series.values

    # do soft majority vote
    columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    df_final = pd.DataFrame(columns=columns)
    for subject_session, subject_df in performance_df.groupby(['participant_id', 'session_id']):
        subject, session = subject_session
        num_patch = len(subject_df.predicted_label)
        p0_all = 0
        p1_all = 0
        # reindex the subject_df.probability
        proba0_series_reindex = subject_df.proba0.reset_index()
        proba1_series_reindex = subject_df.proba1.reset_index()
        weight_series_reindex = subject_df.weight.reset_index()
        y_series_reindex = subject_df.true_label.reset_index()
        y = y_series_reindex.true_label[0]

        for i in range(num_patch):

            p0 = weight_series_reindex.weight[i] * float(proba0_series_reindex.proba0[i])
            p1 = weight_series_reindex.weight[i] * float(proba1_series_reindex.proba1[i])

            p0_all += p0
            p1_all += p1

        proba_list = [p0_all, p1_all]
        y_hat = proba_list.index(max(proba_list))

        row_array = np.array(list([subject, session, y, y_hat])).reshape(1, 4)
        row_df = pd.DataFrame(row_array, columns=columns)
        df_final = df_final.append(row_df)

    results = evaluate_prediction(df_final.true_label.values.astype(int),
                                  df_final.predicted_label.values.astype(int))
    del results['confusion_matrix']

    return df_final, results

#################################
# Datasets
#################################


class MRIDataset_patch(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, caps_directory, data_file, patch_size, stride_size, transformations=None):
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

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")
        participant_list = list(self.df['participant_id'])
        session_list = list(self.df['session_id'])
        label_list = list(self.df['diagnosis'])

        # dynamically calculate the number of patches from each MRI based on the parameters of patch_size & stride_size:
        # Question posted on: https://discuss.pytorch.org/t/how-to-extract-smaller-image-patches-3d/16837/9
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
        # image without intensity normalization
        label = self.diagnosis_code[img_label]
        index_patch = idx % self.patchs_per_patient

        patch_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1', 'preprocessing_dl',
                                  img_name + '_' + sess_name + '_space-MNI_res-1x1x1_patchsize-' + str(self.patch_size)
                                  + '_stride-' + str(self.stride_size) + '_patch-' + str(index_patch) + '.pt')

        patch = torch.load(patch_path)

        # check if the patch has NAN value
        if torch.isnan(patch).any() == True:
            print("Double check, this patch has Nan value: %s" % str(img_name + '_' + sess_name + str(index_patch)))
            patch[torch.isnan(patch)] = 0

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image_id': img_name + '_' + sess_name + '_patch' + str(index_patch), 'image': patch, 'label': label,
                  'participant_id': img_name, 'session_id': sess_name, 'patch_id': index_patch}

        return sample


class MRIDataset_patch_hippocampus(Dataset):
    """Loading the left and right hippocampus ROIs."""

    def __init__(self, caps_directory, data_file, transformations=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transformations (callable, optional): Optional transformations to be applied on a sample.

        """
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")
        participant_list = list(self.df['participant_id'])
        session_list = list(self.df['session_id'])
        label_list = list(self.df['diagnosis'])

        self.patchs_per_patient = 2
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
        ## odd is left hipp, even is right
        left_is_odd = idx % self.patchs_per_patient

        if left_is_odd == 1:
            patch_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1',
                                      'preprocessing_dl',
                                      img_name + '_' + sess_name + '_space-MNI_res-1x1x1_hippocampus_hemi-left.pt')
        else:
            patch_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1',
                                      'preprocessing_dl',
                                      img_name + '_' + sess_name + '_space-MNI_res-1x1x1_hippocampus_hemi-right.pt')

        patch = torch.load(patch_path)

        # check if the patch has NAN value
        if torch.isnan(patch).any():
            print("Double check, this patch has Nan value: %s" % str(img_name + '_' + sess_name + str(left_is_odd)))
            patch[torch.isnan(patch)] = 0

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image_id': img_name + '_' + sess_name + '_patch' + str(left_is_odd), 'image': patch, 'label': label,
                  'participant_id': img_name, 'session_id': sess_name, 'patch_id': left_is_odd}

        return sample


class MRIDataset_patch_by_index(Dataset):
    """Loading the left and right hippocampus ROIs."""

    def __init__(self, caps_directory, data_file, patch_size, stride_size, index_patch, transformations=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transformations (callable, optional): Optional transformations to be applied on a sample.

        """
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.index_patch = index_patch
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}
        self.patch_size = patch_size
        self.stride_size = stride_size

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")
        participant_list = list(self.df['participant_id'])
        session_list = list(self.df['session_id'])
        label_list = list(self.df['diagnosis'])

        self.patch_participant_list = participant_list
        self.patch_session_list = session_list
        self.patch_label_list = label_list

    def __len__(self):
        return len(self.patch_participant_list)

    def __getitem__(self, idx):
        img_name = self.patch_participant_list[idx]
        sess_name = self.patch_session_list[idx]
        img_label = self.patch_label_list[idx]
        label = self.diagnosis_code[img_label]

        patch_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1',
                                  'preprocessing_dl',
                                  img_name + '_' + sess_name + '_space-MNI_res-1x1x1_patchsize-' + str(self.patch_size) + '_stride-' + str(self.stride_size) + '_patch-' + str(
                                      self.index_patch) + '.pt')

        patch = torch.load(patch_path)

        # check if the patch has NAN value
        if torch.isnan(patch).any() == True:
            print("Double check, this patch has Nan value: %s" % str(img_name + '_' + sess_name + str(self.index_patch)))
            patch[torch.isnan(patch)] = 0

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image_id': img_name + '_' + sess_name + '_patch' + str(self.index_patch), 'image': patch, 'label': label,
                  'participant_id': img_name, 'session_id': sess_name, 'patch_id': self.index_patch}

        return sample


def visualize_ae(ae, data, results_path):
    """
    To reconstruct one example batch and save it in nifti format for visualization
    :param ae:
    :param data: tensor, shape [1, 1, height, width, length]
    :param results_path:
    :return:
    """
    import nibabel as nib
    import os

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # set the model to be eval
    ae.eval()
    output = ae(data)
    reconstructed_nii = nib.Nifti1Image(output[0][0].cpu().detach().numpy(), np.eye(4))
    input_nii = nib.Nifti1Image(data[0][0].cpu().detach().numpy(), np.eye(4))
    nib.save(reconstructed_nii, os.path.join(results_path, 'example_patch_reconstructed.nii.gz'))
    nib.save(input_nii, os.path.join(results_path, 'example_patch_original.nii.gz'))
