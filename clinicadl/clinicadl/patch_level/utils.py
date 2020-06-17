# coding: utf8

import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from time import time


#################################
# AutoEncoder train / test
#################################


def stacked_ae_learning(model, train_loader, valid_loader, criterion, writer_train, writer_valid, options, model_dir):
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
    from ..tools.deep_learning.models import AutoEncoder
    from ..tools.deep_learning import save_checkpoint, load_model
    from copy import deepcopy

    # if the model defined is not already constructed to an AE, then we convert the CNN into an AE
    ae = AutoEncoder(model)

    ae_finetuning(ae, train_loader, valid_loader, criterion, writer_train, writer_valid, options, model_dir)

    # Updating and setting weights of the convolutional layers
    checkpoint_dir = path.join(model_dir, 'best_loss')
    best_autodecoder, best_epoch = load_model(ae, checkpoint_dir, options.gpu,  filename='model_best.pth.tar')

    del ae

    # save the encoder part of the AEs, the best AEs has been saved in the ae_finetuning part
    model.features = deepcopy(best_autodecoder.encoder)
    save_checkpoint({'model': model.state_dict(),
                     'epoch': best_epoch},
                    False, False,
                    os.path.join(model_dir, 'Encoder'),
                    filename='model_best_encoder.pth.tar')

    del best_epoch

    return model, best_autodecoder


def ae_finetuning(auto_encoder_all, train_loader, valid_loader, criterion, writer_train_ft, writer_valid_ft, options,
                  model_dir, global_step=0):
    from ..tools.deep_learning import save_checkpoint

    auto_encoder_all.train()
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, auto_encoder_all.parameters()),
                                                         options.learning_rate)
    if options.gpu:
        auto_encoder_all.cuda()

    # Initialize variables
    best_loss_valid = np.inf
    print("Beginning fine-tuning")

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
        loss_valid = test_ae(auto_encoder_all, valid_loader, options.gpu, criterion)
        mean_loss_valid = loss_valid / (len(valid_loader))
        writer_valid_ft.add_scalar('loss', mean_loss_valid, global_step)
        print("Mean validation loss is %f for the -th batch %d" % (mean_loss_valid, global_step))

        # reset the model to train mode after evaluation
        auto_encoder_all.train()

        is_best_loss = loss_valid < best_loss_valid
        # Save best based on smallest loss
        best_loss_valid = min(loss_valid, best_loss_valid)
        save_checkpoint({'model': auto_encoder_all.state_dict(),
                         'epoch': epoch,
                         'valid_loss': mean_loss_valid},
                        False, is_best_loss,
                        model_dir)

    del optimizer, auto_encoder_all


def test_ae(model, dataloader, gpu, criterion):
    """
    Computes the loss of the model, either the loss of the layer-wise AE or all the AEs in a big graph one time.

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param gpu: (bool) if True a gpu is used
    :param criterion:
    :return: loss of the model (float)
    """
    model.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if gpu:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        hidden = inputs
        outputs = model(hidden)
        hidden_requires_grad_no = hidden.detach()
        hidden_requires_grad_no.requires_grad = False
        loss = criterion(outputs, hidden_requires_grad_no)
        total_loss += loss.item()
        torch.cuda.empty_cache()

        del inputs, outputs, loss

    return total_loss


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


#################################
# Transfer learning
#################################

def load_model_after_ae(model, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    Load and copy the weights and biases of a previously trained Encoder part of an autoencoder.

    :param model: (nn.Module) the object in which the weights and biases are copied.
    :param checkpoint_dir: (str) path to the directory in which the pretrained Autoencoder is saved.
    :param filename: (str) name of the file in which the pretrained Autoencoder is saved.
    :return:
        - model_updated (nn.Module) model initialized with the pretrained CNN
        - best_epoch (int) the number of the epoch at which the pretrained CNN corresponds
    """
    from copy import deepcopy

    model_after_ae = deepcopy(model)
    model_dict = model_after_ae.state_dict()
    param_dict = torch.load(os.path.join(checkpoint_dir, filename))
    ae_pretrained_dict = param_dict['model']
    ae_pretrained_dict_copy = deepcopy(ae_pretrained_dict)

    # remove the classifier's weight, only take the convolutional part.
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
    Load and copy the weights and biases of a previously trained CNN.

    :param model: (nn.Module) the object in which the weights and biases are copied.
    :param checkpoint_dir: (str) path to the directory in which the pretrained CNN is saved.
    :param filename: (str) name of the file in which the pretrained CNN is saved.
    :return:
        - model_updated (nn.Module) model initialized with the pretrained CNN
        - best_epoch (int) the number of the epoch at which the pretrained CNN corresponds
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

def train(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch, model_mode="train",
          selection_threshold=None):
    """
    This is the function to train, validate or test the model, depending on the model_mode parameter.
    :param model:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch:
    :return:
    """
    global_step = None
    softmax = torch.nn.Softmax(dim=1)

    if model_mode == "train":
        columns = ['participant_id', 'session_id', 'patch_id', 'true_label', 'predicted_label', 'proba0', 'proba1']
        results_batch_df = pd.DataFrame(columns=columns)
        total_loss = 0.0

        model.train()  # set the model to training mode

        for i, data in enumerate(data_loader):
            # update the global steps
            global_step = i + epoch * len(data_loader)

            if use_cuda:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            gound_truth_list = labels.data.cpu().numpy().tolist()

            output = model(imgs)
            normalized_output = softmax(output)
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
                row = [sub, data['session_id'][idx], data['patch_id'][idx].item(),
                       labels[idx].item(), predicted[idx].item(),
                       normalized_output[idx, 0].item(), normalized_output[idx, 1].item()]
                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_batch_df = pd.concat([results_batch_df, row_df])

            # delete the temporary variables taking the GPU memory
            del imgs, labels, output, predicted, batch_loss, batch_accuracy
            torch.cuda.empty_cache()

        results_batch_df.reset_index(inplace=True, drop=True)
        epoch_metrics = evaluate_prediction(results_batch_df.true_label.values.astype(int),
                                            results_batch_df.predicted_label.values.astype(int))
        accuracy_batch_mean = epoch_metrics['balanced_accuracy']
        loss_batch_mean = total_loss / len(data_loader)
        torch.cuda.empty_cache()

    elif model_mode == "valid":
        results_batch_df, metrics_batch = test(model, data_loader, use_cuda, loss_func)

        # calculate the balanced accuracy
        _, metrics_subject = soft_voting(results_batch_df, results_batch_df, selection_threshold=selection_threshold)
        accuracy_batch_mean = metrics_subject['balanced_accuracy']
        total_loss = metrics_batch['total_loss']
        loss_batch_mean = total_loss / len(data_loader)

        writer.add_scalar('classification accuracy', accuracy_batch_mean, epoch)
        writer.add_scalar('loss', loss_batch_mean, epoch)

        torch.cuda.empty_cache()

    else:
        raise ValueError('This mode %s was not implemented. Please choose between train and valid' % model_mode)

    return results_batch_df, accuracy_batch_mean, loss_batch_mean, global_step


def test(model, dataloader, use_cuda, criterion):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :param criterion: (loss) function to calculate the loss
    :return:
        (DataFrame) results of each session
        (dict) ensemble of metrics + total loss
    """

    softmax = torch.nn.Softmax(dim=1)
    columns = ['participant_id', 'session_id', 'patch_id', 'true_label', 'predicted_label', 'proba0', 'proba1']
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    if use_cuda:
        model.cuda()

    model.eval()  # set the model to evaluation mode
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if use_cuda:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            output = model(imgs)
            normalized_output = softmax(output)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [[sub, data['session_id'][idx], data['patch_id'][idx].item(),
                       labels[idx].item(), predicted[idx].item(),
                       normalized_output[idx, 0].item(), normalized_output[idx, 1].item()]]

                row_df = pd.DataFrame(row, columns=columns)
                results_df = pd.concat([results_df, row_df])

            del imgs, labels, output
            torch.cuda.empty_cache()

        # calculate the balanced accuracy
        results = evaluate_prediction(results_df.true_label.values.astype(int),
                                      results_df.predicted_label.values.astype(int))
        results_df.reset_index(inplace=True, drop=True)
        results['total_loss'] = total_loss
        torch.cuda.empty_cache()

    return results_df, results


def evaluate_prediction(y, y_hat):
    """
    This is a function to calculate the different metrics based on the list of true label and predicted label
    :param y: list of labels
    :param y_hat: list of predictions
    :return: (dict) ensemble of metrics
    """
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
    Save the outputs of the test function to tsv files.

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

    results_df.to_csv(os.path.join(performance_dir, dataset + '_patch_level_result.tsv'), index=False,
                      sep='\t')

    del results['confusion_matrix']
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(performance_dir, dataset + '_patch_level_metrics.tsv'),
                                            index=False, sep='\t')


def retrieve_patch_level_results(output_dir, fold, selection, dataset, num_cnn):
    """Retrieve performance_df for single or multi-CNN framework."""
    if num_cnn is None:
        result_tsv = os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection,
                                  dataset + '_patch_level_result.tsv')
        performance_df = pd.read_csv(result_tsv, sep='\t')

    else:
        performance_df = pd.DataFrame()
        for cnn in range(num_cnn):
            tsv_path = os.path.join(output_dir, 'performances', 'fold_%i' % fold, 'cnn-%i' % cnn, selection,
                                    dataset + '_patch_level_result.tsv')
            cnn_df = pd.read_csv(tsv_path, sep='\t')
            performance_df = pd.concat([performance_df, cnn_df])
        performance_df.reset_index(drop=True, inplace=True)

    return performance_df


def soft_voting_to_tsvs(output_dir, fold, selection, dataset='test', num_cnn=None, selection_threshold=None):
    """
    Save soft voting results to tsv files.

    :param output_dir: (str) path to the output directory.
    :param fold: (int) Fold number of the cross-validation.
    :param selection: (str) criterion on which the model is selected (either best_loss or best_acc)
    :param dataset: (str) name of the dataset for which the soft-voting is performed. If different from training or
                    validation, the weights of soft voting will be computed on validation accuracies.
    :param num_cnn: (int) if given load the patch level results of a multi-CNN framework.
    :param selection_threshold: (float) all patches for which the classification accuracy is below the
                                threshold is removed.

    """

    # Choose which dataset is used to compute the weights of soft voting.
    if dataset in ['train', 'validation']:
        validation_dataset = dataset
    else:
        validation_dataset = 'validation'
    test_df = retrieve_patch_level_results(output_dir, fold, selection, dataset, num_cnn)
    validation_df = retrieve_patch_level_results(output_dir, fold, selection, validation_dataset, num_cnn)

    performance_path = os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection)
    if not os.path.exists(performance_path):
        os.makedirs(performance_path)

    df_final, metrics = soft_voting(test_df, validation_df, selection_threshold=selection_threshold)

    df_final.to_csv(os.path.join(os.path.join(performance_path, dataset + '_image_level_result.tsv')),
                    index=False, sep='\t')

    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection,
                                                         dataset + '_image_level_metrics.tsv'),
                                            index=False, sep='\t')


def soft_voting(performance_df, validation_df, selection_threshold=None):
    """
    Computes soft voting based on the probabilities in performance_df. Weights are computed based on the accuracies
    of validation_df.

    ref: S. Raschka. Python Machine Learning., 2015
    :param performance_df: (DataFrame) results on patch level of the set on which the combination is made.
    :param validation_df: (DataFrame) results on patch level of the set used to compute the weights.
    :param selection_threshold: (float) if given, all patches for which the classification accuracy is below the
                                threshold is removed.
    :return:
        - df_final (DataFrame) the results on the subject level
        - results (dict) the metrics on the subject level
    """

    # Compute the patch accuracies on the validation set:
    validation_df["accurate_prediction"] = validation_df.apply(lambda x: check_prediction(x), axis=1)
    patch_accuracies = validation_df.groupby("patch_id")["accurate_prediction"].sum()
    if selection_threshold is not None:
        patch_accuracies[patch_accuracies < selection_threshold] = 0
    weight_series = patch_accuracies / patch_accuracies.sum()

    # Sort patches to allow weighted average computation
    performance_df.sort_values(['participant_id', 'session_id', 'patch_id'], inplace=True)
    weight_series.sort_index(inplace=True)

    # Soft majority vote
    columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    df_final = pd.DataFrame(columns=columns)
    for (subject, session), subject_df in performance_df.groupby(['participant_id', 'session_id']):
        y = subject_df["true_label"].unique().item()
        proba0 = np.average(subject_df["proba0"], weights=weight_series)
        proba1 = np.average(subject_df["proba1"], weights=weight_series)
        proba_list = [proba0, proba1]
        y_hat = proba_list.index(max(proba_list))

        row = [[subject, session, y, y_hat]]
        row_df = pd.DataFrame(row, columns=columns)
        df_final = df_final.append(row_df)

    results = evaluate_prediction(df_final.true_label.values.astype(int),
                                  df_final.predicted_label.values.astype(int))
    del results['confusion_matrix']

    return df_final, results


def check_prediction(row):
    if row["true_label"] == row["predicted_label"]:
        return 1
    else:
        return 0
