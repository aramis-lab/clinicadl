from __future__ import print_function
import torch
import numpy as np
import os
import warnings
import pandas as pd
from time import time

from clinicadl.tools.deep_learning.iotools import check_and_clean, visualize_subject 
from clinicadl.tools.deep_learning import EarlyStopping, save_checkpoint


#####################
# CNN train / test  #
#####################

def train(model, train_loader, valid_loader, criterion, optimizer, resume, options):
    """
    Function used to train a CNN.
    The best model and checkpoint will be found in the 'best_model_dir' of options.output_dir.

    :param model: (Module) CNN to be trained
    :param train_loader: (DataLoader) wrapper of the training dataset
    :param valid_loader: (DataLoader) wrapper of the validation dataset
    :param criterion: (loss) function to calculate the loss
    :param optimizer: (torch.optim) optimizer linked to model parameters
    :param resume: (bool) if True, a begun job is resumed
    :param options: (Namespace) ensemble of other options given to the main script.
    """
    from tensorboardX import SummaryWriter
    from time import time

    columns = ['epoch', 'iteration', 'acc_train', 'mean_loss_train', 'acc_valid', 'mean_loss_valid', 'time']
    log_dir = os.path.join(options.output_dir, 'log_dir', 'fold_' + str(options.split), 'CNN')
    best_model_dir = os.path.join(options.output_dir, 'best_model_dir', 'fold_' + str(options.split), 'CNN')
    filename = os.path.join(log_dir, 'training.tsv')

    if not resume:
        check_and_clean(best_model_dir)
        check_and_clean(log_dir)

        results_df = pd.DataFrame(columns=columns)
        with open(filename, 'w') as f:
            results_df.to_csv(f, index=False, sep='\t')
        options.beginning_epoch = 0

    else:
        if not os.path.exists(filename):
            raise ValueError('The training.tsv file of the resumed experiment does not exist.')
        truncated_tsv = pd.read_csv(filename, sep='\t')
        truncated_tsv.set_index(['epoch', 'iteration'], inplace=True)
        truncated_tsv.drop(options.beginning_epoch, level=0, inplace=True)
        truncated_tsv.to_csv(filename, index=True, sep='\t')

    # Create writers
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_valid = SummaryWriter(os.path.join(log_dir, 'valid'))

    # Initialize variables
    best_valid_accuracy = 0.0
    best_valid_loss = np.inf
    epoch = options.beginning_epoch

    model.train()  # set the module to training mode

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    mean_loss_valid = None
    t_beggining = time()

    while epoch < options.epochs and not early_stopping.step(mean_loss_valid):
        print("At %d-th epoch." % epoch)

        model.zero_grad()
        evaluation_flag = True
        step_flag = True
        tend = time()
        total_time = 0

        for i, data in enumerate(train_loader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if options.gpu:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']
            train_output = model(imgs)
            _, predict_batch = train_output.topk(1)
            loss = criterion(train_output, labels)

            # Back propagation
            loss.backward()

            del imgs, labels

            if (i+1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                del loss

                # Evaluate the model only when no gradients are accumulated
                if(i+1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)

                    acc_mean_train, total_loss_train = test(model, train_loader, options.gpu, criterion)
                    mean_loss_train = total_loss_train / (len(train_loader) * train_loader.batch_size)

                    acc_mean_valid, total_loss_valid = test(model, valid_loader, options.gpu, criterion)
                    mean_loss_valid = total_loss_valid / (len(valid_loader) * valid_loader.batch_size)
                    model.train()

                    writer_train.add_scalar('balanced_accuracy', acc_mean_train, i + epoch * len(train_loader))
                    writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
                    writer_valid.add_scalar('balanced_accuracy', acc_mean_valid, i + epoch * len(train_loader))
                    writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
                    print("Subject level training accuracy is %f at the end of iteration %d" % (acc_mean_train, i))
                    print("Subject level validation accuracy is %f at the end of iteration %d" % (acc_mean_valid, i))

                    t_current = time() - t_beggining
                    row = np.array([epoch, i, acc_mean_train, mean_loss_train, acc_mean_valid, mean_loss_valid,
                                    t_current]).reshape(1, -1)
                    row_df = pd.DataFrame(row, columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

            tend = time()
        print('Mean time per batch (train):', total_time / len(train_loader) * train_loader.batch_size)

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        elif evaluation_flag:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        model.zero_grad()
        print('Last checkpoint at the end of the epoch %d' % epoch)

        acc_mean_train, total_loss_train = test(model, train_loader, options.gpu, criterion)
        mean_loss_train = total_loss_train / (len(train_loader) * train_loader.batch_size)

        acc_mean_valid, total_loss_valid = test(model, valid_loader, options.gpu, criterion)
        mean_loss_valid = total_loss_valid / (len(valid_loader) * valid_loader.batch_size)
        model.train()

        writer_train.add_scalar('balanced_accuracy', acc_mean_train, i + epoch * len(train_loader))
        writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
        writer_valid.add_scalar('balanced_accuracy', acc_mean_valid, i + epoch * len(train_loader))
        writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
        print("Subject level training accuracy is %f at the end of iteration %d" % (acc_mean_train, i))
        print("Subject level validation accuracy is %f at the end of iteration %d" % (acc_mean_valid, i))

        t_current = time() - t_beggining
        row = np.array([epoch, i, acc_mean_train, mean_loss_train, acc_mean_valid, mean_loss_valid,
                        t_current]).reshape(1, -1)
        row_df = pd.DataFrame(row, columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=False, index=False, sep='\t')
        accuracy_is_best = acc_mean_valid > best_valid_accuracy
        loss_is_best = mean_loss_valid < best_valid_loss
        best_valid_accuracy = max(acc_mean_valid, best_valid_accuracy)
        best_valid_loss = min(mean_loss_valid, best_valid_loss)

        save_checkpoint({'model': model.state_dict(),
                         'epoch': epoch,
                         'valid_acc': acc_mean_valid},
                        accuracy_is_best, loss_is_best,
                        best_model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': options.optimizer,
                         },
                        False, False,
                        best_model_dir,
                        filename='optimizer.pth.tar')

        epoch += 1


def evaluate_prediction(y, y_pred):

    """
    This is a function to calculate the different metrics based on the list of true label and predicted label
    :param y: list of labels
    :param y_pred: list of predictions
    :return: (dict) ensemble of metrics
    """

    true_positive = np.sum((y_pred == 1) & (y == 1))
    true_negative = np.sum((y_pred == 0) & (y == 0))
    false_positive = np.sum((y_pred == 1) & (y == 0))
    false_negative = np.sum((y_pred == 0) & (y == 1))

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
               }

    return results


def test(model, dataloader, use_cuda, criterion, full_return=False):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :param criterion: (loss) function to calculate the loss
    :param full_return: if True also returns the sensitivities and specificities for a multiclass problem
    :return:
    if full_return
        (dict) ensemble of metrics
        (float) total loss
        (DataFrame) results of each session
    else
        (float) balanced accuracy
        (float) total loss

    """
    model.eval()

    columns = ["participant_id", "session_id", "true_label", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)

    total_time = 0
    total_loss = 0
    tend = time()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if use_cuda:
                inputs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                inputs, labels = data['image'], data['label']
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx], labels[idx].item(), predicted[idx].item()]
                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            del inputs, outputs, labels, loss
            tend = time()
        print('Mean time per batch (test):', total_time / len(dataloader) * dataloader.batch_size)
        results_df.reset_index(inplace=True, drop=True)

        results = evaluate_prediction(results_df.true_label.values.astype(int),
                                      results_df.predicted_label.values.astype(int))

    if full_return:
        return results, total_loss, results_df

    return results['balanced_accuracy'], total_loss


#############################
# AutoEncoder train / test  #
#############################

def ae_finetuning(decoder, train_loader, valid_loader, criterion, optimizer, resume, options):
    """
    Function used to train an autoencoder.
    The best autoencoder and checkpoint will be found in the 'best_model_dir' of options.output_dir.

    :param decoder: (Autoencoder) Autoencoder constructed from a CNN with the Autoencoder class.
    :param train_loader: (DataLoader) wrapper of the training dataset.
    :param valid_loader: (DataLoader) wrapper of the validation dataset.
    :param criterion: (loss) function to calculate the loss.
    :param optimizer: (torch.optim) optimizer linked to model parameters.
    :param resume: (bool) if True, a begun job is resumed.
    :param options: (Namespace) ensemble of other options given to the main script.
    """
    from tensorboardX import SummaryWriter

    log_dir = os.path.join(options.output_dir, 'log_dir', 'fold_' + str(options.split), 'ConvAutoencoder')
    visualization_path = os.path.join(options.output_dir, 'visualize', 'fold_' + str(options.split))
    best_model_dir = os.path.join(options.output_dir, 'best_model_dir', 'fold_' + str(options.split), 'ConvAutoencoder')
    filename = os.path.join(log_dir, 'training.tsv')

    if not resume:
        check_and_clean(best_model_dir)
        check_and_clean(visualization_path)
        check_and_clean(log_dir)
        columns = ['epoch', 'iteration', 'loss_train', 'mean_loss_train', 'loss_valid', 'mean_loss_valid']
        results_df = pd.DataFrame(columns=columns)
        with open(filename, 'w') as f:
            results_df.to_csv(f, index=False, sep='\t')
        options.beginning_epoch = 0

    else:
        if not os.path.exists(filename):
            raise ValueError('The training.tsv file of the resumed experiment does not exist.')
        truncated_tsv = pd.read_csv(filename, sep='\t')
        truncated_tsv.set_index(['epoch', 'iteration'], inplace=True)
        truncated_tsv.drop(options.beginning_epoch, level=0, inplace=True)
        truncated_tsv.to_csv(filename, index=True, sep='\t')

    # Create writers
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_valid = SummaryWriter(os.path.join(log_dir, 'valid'))

    decoder.train()
    first_visu = True
    print(decoder)

    if options.gpu:
        decoder.cuda()

    # Initialize variables
    best_loss_valid = np.inf

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    loss_valid = None
    epoch = options.beginning_epoch

    print("Beginning training")
    while epoch < options.epochs and not early_stopping.step(loss_valid):
        print("At %d-th epoch." % epoch)

        decoder.zero_grad()
        evaluation_flag = True
        step_flag = True
        for i, data in enumerate(train_loader):
            if options.gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']

            train_output = decoder(imgs)
            loss = criterion(train_output, imgs)
            loss.backward()

            del imgs, train_output

            if (i+1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                # Evaluate the decoder only when no gradients are accumulated
                if (i+1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)
                    loss_train = test_ae(decoder, train_loader, options.gpu, criterion)
                    mean_loss_train = loss_train / (len(train_loader) * train_loader.batch_size)

                    loss_valid = test_ae(decoder, valid_loader, options.gpu, criterion)
                    mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.batch_size)
                    decoder.train()

                    writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
                    writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
                    print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))
                    row = np.array([epoch, i, loss_train, mean_loss_train, loss_valid, mean_loss_valid]).reshape(1, -1)
                    row_df = pd.DataFrame(row, columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        if evaluation_flag:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        print('Last checkpoint at the end of the epoch %d' % epoch)

        loss_train = test_ae(decoder, train_loader, options.gpu, criterion)
        mean_loss_train = loss_train / (len(train_loader) * train_loader.batch_size)

        loss_valid = test_ae(decoder, valid_loader, options.gpu, criterion)
        mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.batch_size)
        decoder.train()

        writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
        writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
        print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

        row = np.array([epoch, i, loss_train, mean_loss_train, loss_valid, mean_loss_valid]).reshape(1, -1)
        row_df = pd.DataFrame(row, columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=False, index=False, sep='\t')

        is_best = loss_valid < best_loss_valid
        best_loss_valid = min(best_loss_valid, loss_valid)
        # Always save the model at the end of the epoch and update best model
        save_checkpoint({'model': decoder.state_dict(),
                         'iteration': i,
                         'epoch': epoch,
                         'loss_valid': loss_valid},
                        False, is_best,
                        best_model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': options.optimizer,
                         },
                        False, False,
                        best_model_dir,
                        filename='optimizer.pth.tar')

        if epoch % 10 == 0:
            visualize_subject(decoder, train_loader, visualization_path, options, epoch=epoch, save_input=first_visu)
            first_visu = False

        epoch += 1

    visualize_subject(decoder, train_loader, visualization_path, options, epoch=epoch, save_input=first_visu)


def test_ae(model, dataloader, use_cuda, criterion):
    """
    Computes the loss of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :param criterion: (loss) function to calculate the loss
    :return: loss of the model (float)
    """
    model.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()

        del inputs, outputs, loss

    return total_loss
