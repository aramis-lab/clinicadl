from __future__ import print_function
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import shutil
import warnings
import pandas as pd
from time import time


def train(model, train_loader, valid_loader, criterion, optimizer, resume, options):
    """
    This is the function to train the model
    :param model:
    :param train_loader:
    :param valid_loader:
    :param criterion:
    :param optimizer:
    :param options:
    """
    from tensorboardX import SummaryWriter
    from time import time

    columns = ['epoch', 'iteration', 'acc_train', 'mean_loss_train', 'acc_valid', 'mean_loss_valid', 'time']
    log_dir = os.path.join(options.output_dir, 'log_dir', 'CNN', 'fold_' + str(options.split))
    best_model_dir = os.path.join(options.output_dir, 'best_model_dir', 'CNN', 'fold_' + str(options.split))
    filename = os.path.join(log_dir, 'training.tsv')

    if not resume:
        check_and_clean(log_dir)
        check_and_clean(best_model_dir)

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

        # Initialize metrics for training evaluation
        concat_loss = []
        if options.gpu:
            concat_prediction = torch.LongTensor().cuda()
            concat_truth = torch.LongTensor().cuda()
        else:
            concat_prediction = torch.LongTensor()
            concat_truth = torch.LongTensor()

        for i, data in enumerate(train_loader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if options.gpu:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']
            train_output = model(imgs)
            _, predict_batch = train_output.topk(1)
            concat_prediction = torch.cat((concat_prediction, predict_batch), 0)
            concat_truth = torch.cat((concat_truth, labels))
            loss = criterion(train_output, labels)
            concat_loss.append(loss.item())

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
                    # Training performance is evaluated on the last batches seen
                    if options.gpu:
                        concat_prediction_arr = concat_prediction.view(-1).cpu().numpy().astype(int)
                        concat_truth_arr = concat_truth.view(-1).cpu().numpy().astype(int)
                    else:
                        concat_prediction_arr = concat_prediction.view(-1).numpy().astype(int)
                        concat_truth_arr = concat_truth.view(-1).numpy().astype(int)

                    if options.training_evaluation == 'n_batches':
                        metrics = evaluate_prediction(concat_truth_arr, concat_prediction_arr, options.evaluation_steps)
                        acc_mean_train = metrics['balanced_accuracy']
                        total_loss_train = sum(concat_loss[-options.evaluation_steps:])
                        mean_loss_train = total_loss_train / (options.evaluation_steps * train_loader.batch_size)
                    else:
                        acc_mean_train, total_loss_train = test(model, train_loader, options.gpu, criterion)
                        mean_loss_train = total_loss_train / (len(train_loader) * train_loader.batch_size)

                    acc_mean_valid, total_loss_valid = test(model, valid_loader, options.gpu, criterion)
                    mean_loss_valid = total_loss_valid / (len(valid_loader) * valid_loader.batch_size)
                    model.train()

                    writer_train.add_scalar('balanced_accuracy', acc_mean_train, i + epoch * len(train_loader))
                    writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
                    writer_valid.add_scalar('balanced_accuracy', acc_mean_valid, i + epoch * len(train_loader))
                    writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
                    print("Scan level training accuracy is %f at the end of iteration %d" % (acc_mean_train, i))
                    print("Scan level validation accuracy is %f at the end of iteration %d" % (acc_mean_valid, i))

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
        # Training performance is evaluated on the last batches seen
        if options.gpu:
            concat_prediction_arr = concat_prediction.view(-1).cpu().numpy().astype(int)
            concat_truth_arr = concat_truth.view(-1).cpu().numpy().astype(int)
        else:
            concat_prediction_arr = concat_prediction.view(-1).numpy().astype(int)
            concat_truth_arr = concat_truth.numpy().view(-1).astype(int)

        if options.training_evaluation == 'n_batches':
            metrics = evaluate_prediction(concat_truth_arr, concat_prediction_arr, options.evaluation_steps)
            acc_mean_train = metrics['balanced_accuracy']
            total_loss_train = sum(concat_loss[-options.evaluation_steps:])
            mean_loss_train = total_loss_train / (options.evaluation_steps * train_loader.batch_size)
        else:
            acc_mean_train, total_loss_train = test(model, train_loader, options.gpu, criterion)
            mean_loss_train = total_loss_train / (len(train_loader) * train_loader.batch_size)

        acc_mean_valid, total_loss_valid = test(model, valid_loader, options.gpu, criterion)
        mean_loss_valid = total_loss_valid / (len(valid_loader) * valid_loader.batch_size)
        model.train()

        writer_train.add_scalar('balanced_accuracy', acc_mean_train, i + epoch * len(train_loader))
        writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
        writer_valid.add_scalar('balanced_accuracy', acc_mean_valid, i + epoch * len(train_loader))
        writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
        print("Scan level training accuracy is %f at the end of iteration %d" % (acc_mean_train, i))
        print("Scan level validation accuracy is %f at the end of iteration %d" % (acc_mean_valid, i))

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


def evaluate_prediction(concat_true, concat_prediction, horizon=None):

    """
    This is a function to calculate the different metrics based on the list of true label and predicted label
    :param concat_true: list of concatenated last labels
    :param concat_prediction: list of concatenated last prediction
    :param horizon: (int) number of batches to consider to evaluate performance
    :return:
    """

    if horizon is not None:
        y = list(concat_true)[-horizon:]
        y_hat = list(concat_prediction)[-horizon:]
    else:
        y = list(concat_true)
        y_hat = list(concat_prediction)

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


def test(model, dataloader, use_cuda, criterion, verbose=False, full_return=False):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :param full_return: if True also returns the sensitivities and specificities for a multiclass problem
    :return:
        balanced accuracy of the model (float)
        total loss on the dataloader
    """
    model.eval()

    # Use tensors instead of arrays to avoid bottlenecks
    predicted_tensor = torch.zeros(len(dataloader.dataset))
    truth_tensor = torch.zeros(len(dataloader.dataset))

    columns = ["participant_id", "session_id", "true_label", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)

    if use_cuda:
        predicted_tensor = predicted_tensor.cuda()
        truth_tensor = truth_tensor.cuda()

    total_time = 0
    total_loss = 0
    tend = time()
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

        idx = i * dataloader.batch_size
        idx_end = (i + 1) * dataloader.batch_size
        predicted_tensor[idx:idx_end:] = predicted
        truth_tensor[idx:idx_end:] = labels

        del inputs, outputs, labels, loss
        tend = time()
    print('Mean time per batch (test):', total_time / len(dataloader) * dataloader.batch_size)
    results_df.reset_index(inplace=True, drop=True)

    # Cast to numpy arrays to avoid bottleneck in the next loop
    if use_cuda:
        predicted_arr = predicted_tensor.cpu().numpy().astype(int)
        truth_arr = truth_tensor.cpu().numpy().astype(int)
    else:
        predicted_arr = predicted_tensor.numpy().astype(int)
        truth_arr = truth_tensor.numpy().astype(int)

    results = evaluate_prediction(truth_arr, predicted_arr)

    if full_return:
        return results, total_loss, results_df

    return results['balanced_accuracy'], total_loss


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def save_checkpoint(state, accuracy_is_best, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar',
                    best_accuracy='best_acc', best_loss='best_loss'):

    torch.save(state, os.path.join(checkpoint_dir, filename))
    if accuracy_is_best:
        best_accuracy_path = os.path.join(checkpoint_dir, best_accuracy)
        if not os.path.exists(best_accuracy_path):
            os.makedirs(best_accuracy_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(best_accuracy_path, 'model_best.pth.tar'))

    if loss_is_best:
        best_loss_path = os.path.join(checkpoint_dir, best_loss)
        if not os.path.exists(best_loss_path):
            os.makedirs(best_loss_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(best_loss_path, 'model_best.pth.tar'))


def load_model(model, checkpoint_dir, filename='model_best.pth.tar'):
    from copy import deepcopy

    best_model = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, filename))
    best_model.load_state_dict(param_dict['model'])
    return best_model, param_dict['epoch']


def check_and_clean(d):

    # if os.path.exists(d):
    #     shutil.rmtree(d)
    # os.makedirs(d)
    if not os.path.exists(d):
        os.makedirs(d)


def ae_finetuning(decoder, train_loader, valid_loader, criterion, optimizer, resume, options):
    from tensorboardX import SummaryWriter

    log_dir = os.path.join(options.output_dir, 'log_dir', 'ConvAutoencoder', 'fold_' + str(options.split))
    best_model_dir = os.path.join(options.output_dir, 'best_model_dir', 'ConvAutoencoder', 'fold_' + str(options.split))
    filename = os.path.join(log_dir, 'training.tsv')

    if not resume:
        check_and_clean(log_dir)
        check_and_clean(best_model_dir)
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
    while epoch < options.transfer_learning_epochs and not early_stopping.step(loss_valid):
        print("At %d-th epoch." % epoch)

        decoder.zero_grad()
        evaluation_flag = True
        step_flag = True
        concat_loss = []
        for i, data in enumerate(train_loader):
            if options.gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']

            train_output = decoder(imgs)
            loss = criterion(train_output, imgs)
            concat_loss.append(loss.item())
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
                    if options.training_evaluation == 'n_batches':
                        loss_train = sum(concat_loss[-options.evaluation_steps:])
                        mean_loss_train = loss_train / (options.evaluation_steps * train_loader.batch_size)
                    else:
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

        if options.training_evaluation == 'n_batches':
            loss_train = sum(concat_loss[-options.evaluation_steps:])
            mean_loss_train = loss_train / (options.evaluation_steps * train_loader.batch_size)
        else:
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
            visualize_subject(decoder, train_loader, log_dir, epoch, options, first_visu,
                              data_path=options.preprocessing)
            first_visu = False

        epoch += 1


def test_ae(model, dataloader, use_cuda, criterion, first_layers=None):
    """
    Computes the loss of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :return: loss of the model (float)
    """
    model.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        if first_layers is not None:
            hidden = first_layers(inputs)
        else:
            hidden = inputs
        outputs = model(hidden)
        loss = criterion(outputs, hidden)
        total_loss += loss.item()

        del inputs, outputs, loss

    return total_loss


def greedy_learning(model, train_loader, valid_loader, criterion, optimizer, resume, options):
    from os import path
    from utils.model import Decoder
    from copy import deepcopy

    if resume:
        raise NotImplementedError('The resuming version of greedy learning of AE is not implemented.')

    if not isinstance(model, Decoder):
        decoder = Decoder(model)
    else:
        decoder = deepcopy(model)

    level = 0
    first_layers = extract_first_layers(decoder, level)
    auto_encoder = extract_ae(decoder, level)

    while len(auto_encoder) > 0:
        print('Cell learning level %i' % level)
        level_path = path.join(options.output_dir, 'level-' + str(level))
        # Create the method to train with first layers
        ae_training(auto_encoder, first_layers, train_loader, valid_loader, criterion, level_path, options)
        best_ae, _ = load_model(auto_encoder, level_path)

        # Copy the weights of best_ae in decoder encoder and decoder layers
        set_weights(decoder, best_ae, level)

        # Prepare next iteration
        level += 1
        first_layers = extract_first_layers(decoder, level)
        auto_encoder = extract_ae(decoder, level)

    if options.add_sigmoid:
        if isinstance(decoder.decoder[-1], torch.nn.ReLU):
            decoder.decoder = torch.nn.Sequential(*list(decoder.decoder)[:-1])
        decoder.decoder.add_module("sigmoid", torch.nn.Sigmoid())

    ae_finetuning(decoder, train_loader, valid_loader, criterion, optimizer, False, options)

    # Updating and setting weights of the convolutional layers
    best_decoder, best_epoch = load_model(decoder, options.output_dir)
    if not isinstance(model, Decoder):
        model.features = deepcopy(best_decoder.encoder)
        save_checkpoint({'model': model.state_dict(),
                         'epoch': best_epoch},
                        False, False,
                        os.path.join(options.output_dir),
                        'model_pretrained.pth.tar')

    if options.visualization:
        visualize_ae(best_decoder, train_loader, os.path.join(options.output_dir, "train"), options.gpu)
        visualize_ae(best_decoder, valid_loader, os.path.join(options.output_dir, "valid"), options.gpu)

    return model


def set_weights(decoder, auto_encoder, level):
    import torch.nn as nn

    n_conv = 0
    i_ae = 0

    for i, layer in enumerate(decoder.encoder):
        if isinstance(layer, nn.Conv3d):
            n_conv += 1

        if n_conv == level + 1:
            decoder.encoder[i] = auto_encoder.encoder[i_ae]
            # Do BatchNorm layers are not used in decoder
            if not isinstance(layer, nn.BatchNorm3d):
                decoder.decoder[len(decoder) - (i+1)] = auto_encoder.decoder[len(auto_encoder) - (i_ae+1)]
            i_ae += 1

    return decoder


def ae_training(auto_encoder, first_layers, train_loader, valid_loader, criterion, results_path, options):
    from os import path
    from tensorboardX import SummaryWriter

    if not path.exists(results_path):
        os.makedirs(results_path)

    filename = os.path.join(results_path, 'training.tsv')

    # Create writers
    writer_train = SummaryWriter(os.path.join(results_path, 'train'))
    writer_valid = SummaryWriter(os.path.join(results_path, 'valid'))

    columns = ['epoch', 'iteration', 'loss_train', 'mean_loss_train', 'loss_valid', 'mean_loss_valid']
    results_df = pd.DataFrame(columns=columns)
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')

    auto_encoder.train()
    first_layers.eval()
    print(first_layers)
    print(auto_encoder)
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, auto_encoder.parameters()),
                                                         options.transfer_learning_rate)

    if options.gpu:
        auto_encoder.cuda()

    # Initialize variables
    best_loss_valid = np.inf
    epoch = 0

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    loss_valid = None
    print("Beginning training")

    while epoch < options.transfer_learning_epochs and not early_stopping.step(loss_valid):
        print("At %d-th epoch." % epoch)

        auto_encoder.zero_grad()
        evaluation_flag = True
        step_flag = True
        concat_loss = []
        for i, data in enumerate(train_loader):
            if options.gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']

            hidden = first_layers(imgs)
            train_output = auto_encoder(hidden)
            loss = criterion(train_output, hidden)
            concat_loss.append(loss.item())

            loss.backward()

            # writer_train.add_scalar('training_loss', loss.item() / len(data), i + epoch * len(train_loader.dataset))

            if (i+1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                # Evaluate the decoder only when no gradients are accumulated
                if (i+1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)
                    if options.training_evaluation == 'n_batches':
                        loss_train = sum(concat_loss[-options.evaluation_steps:])
                        mean_loss_train = loss_train / (options.evaluation_steps * train_loader.batch_size)
                    else:
                        loss_train = test_ae(auto_encoder, train_loader, options.gpu, criterion)
                        mean_loss_train = loss_train / (len(train_loader) * train_loader.batch_size)

                    loss_valid = test_ae(auto_encoder, valid_loader, options.gpu, criterion, first_layers=first_layers)
                    mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.dataset.size)
                    auto_encoder.train()

                    writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
                    writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
                    print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

                    row = np.array([epoch, i, loss_train, mean_loss_train, loss_valid, mean_loss_valid]).reshape(1, -1)
                    row_df = pd.DataFrame(row, columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

            del imgs

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        if evaluation_flag:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        print('Last checkpoint at the end of the epoch %d' % epoch)
        if options.training_evaluation == 'n_batches':
            loss_train = sum(concat_loss[-options.evaluation_steps:])
            mean_loss_train = loss_train / (options.evaluation_steps * train_loader.batch_size)
        else:
            loss_train = test_ae(auto_encoder, train_loader, options.gpu, criterion)
            mean_loss_train = loss_train / (len(train_loader) * train_loader.batch_size)

        loss_valid = test_ae(auto_encoder, valid_loader, options.gpu, criterion, first_layers=first_layers)
        mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.dataset.size)
        auto_encoder.train()

        writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
        writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
        print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

        row = np.array([epoch, i, loss_train, mean_loss_train, loss_valid, mean_loss_valid]).reshape(1, -1)
        row_df = pd.DataFrame(row, columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=False, index=False, sep='\t')

        is_best = loss_valid < best_loss_valid
        # Save only if is best to avoid performance deterioration
        if is_best:
            best_loss_valid = loss_valid
            save_checkpoint({'model': auto_encoder.state_dict(),
                             'iteration': i,
                             'epoch': epoch,
                             'loss_valid': loss_valid},
                            False, is_best,
                            results_path)


def extract_ae(decoder, level):
    import torch.nn as nn
    from utils.model import Decoder

    n_conv = 0
    output_decoder = Decoder()
    inverse_layers = []

    for i, layer in enumerate(decoder.encoder):
        if isinstance(layer, nn.Conv3d):
            n_conv += 1

        if n_conv == level + 1:
            output_decoder.encoder.add_module(str(len(output_decoder.encoder)), layer)
            inverse_layers.append(decoder.decoder[len(decoder.decoder) - (i + 1)])

        elif n_conv > level + 1:
            break

    inverse_layers.reverse()
    output_decoder.decoder = nn.Sequential(*inverse_layers)
    return output_decoder


def extract_first_layers(decoder, level):
    import torch.nn as nn
    from copy import deepcopy
    from utils.modules import PadMaxPool3d

    n_conv = 0
    first_layers = nn.Sequential()

    for i, layer in enumerate(decoder.encoder):
        if isinstance(layer, nn.Conv3d):
            n_conv += 1

        if n_conv < level + 1:
            layer_copy = deepcopy(layer)
            layer_copy.requires_grad = False
            if isinstance(layer, PadMaxPool3d):
                layer_copy.set_new_return(False, False)

            first_layers.add_module(str(i), layer_copy)
        else:
            break

    return first_layers


def visualize_subject(decoder, dataloader, results_path, epoch, options, first_time=False, data_path='linear'):
    from os import path
    import nibabel as nib
    from utils.data_utils import MinMaxNormalization

    visualization_path = path.join(results_path, 'iterative_visualization')

    if not path.exists(visualization_path):
        os.makedirs(visualization_path)

    set_df = dataloader.dataset.df
    subject = set_df.loc[0, 'participant_id']
    session = set_df.loc[0, 'session_id']
    if data_path == 'linear':
        image_path = path.join(options.input_dir, 'subjects', subject, session,
                               't1', 'preprocessing_dl',
                               subject + '_' + session + '_space-MNI_res-1x1x1.nii.gz')
    elif data_path == 'mni':
        image_path = path.join(options.input_dir, 'subjects', subject, session,
                               't1', 'spm', 'segmentation', 'normalized_space',
                               subject + '_' + session + '_space-Ixi549Space_T1w.nii.gz')
    else:
        raise NotImplementedError('Data path %s is not implemented' % data_path)

    input_nii = nib.load(image_path)
    input_np = input_nii.get_data().astype(float)
    np.nan_to_num(input_np, copy=False)
    input_pt = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0).float()
    if options.minmaxnormalization:
        transform = MinMaxNormalization()
        input_pt = transform(input_pt)

    if options.gpu:
        input_pt = input_pt.cuda()

    output_pt = decoder(input_pt)

    if options.gpu:
        output_pt = output_pt.cpu()

    output_np = output_pt.detach().numpy()[0][0]
    output_nii = nib.Nifti1Image(output_np, affine=input_nii.affine)

    if first_time:
        nib.save(input_nii, path.join(visualization_path, 'input.nii'))

    nib.save(output_nii, path.join(visualization_path, 'epoch-' + str(epoch) + '.nii'))


def visualize_ae(decoder, dataloader, results_path, gpu, data_path='linear'):
    import nibabel as nib
    from utils.data_utils import ToTensor
    import os
    from os import path

    if not path.exists(results_path):
        os.makedirs(results_path)

    subject = dataloader.dataset.df.loc[0, 'participant_id']
    session = dataloader.dataset.df.loc[0, 'session_id']

    if data_path == 'linear':
        image_path = path.join(dataloader.dataset.img_dir, 'subjects', subject, session,
                               't1', 'preprocessing_dl',
                               subject + '_' + session + '_space-MNI_res-1x1x1.nii.gz')
    elif data_path == 'mni':
        image_path = path.join(dataloader.dataset.img_dir, 'subjects', subject, session,
                               't1', 'spm', 'segmentation', 'normalized_space',
                               subject + '_' + session + '_space-Ixi549Space_T1w.nii.gz')
    else:
        raise NotImplementedError('Data path %s is not implemented' % data_path)

    data = nib.load(image_path)
    img = data.get_data()
    affine = data.get_affine()
    img_tensor = ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    if gpu:
        img_tensor = img_tensor.cuda()
    print(img_tensor.size())
    output_tensor = decoder(img_tensor)
    output = nib.Nifti1Image(output_tensor[0][0].cpu().detach().numpy(), affine)
    nib.save(output, os.path.join(results_path, 'output_image.nii'))
    nib.save(data, os.path.join(results_path, 'input_image.nii'))


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.is_better = lambda a, best: a < best - best * min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + best * min_delta


def commandline_to_json(commandline, model_type):
    """
    This is a function to write the python argparse object into a jason file. This helps for DL when searching for hyperparameters
    :param commandline: a tuple contain the output of `parser.parse_known_args()`
    :return:
    """
    import json

    commandline_arg_dic = vars(commandline[0])
    commandline_arg_dic['unknown_arg'] = commandline[1]

    # if train_from_stop_point, do not delete the folders
    output_dir = commandline_arg_dic['output_dir']
    log_dir = os.path.join(output_dir, 'log_dir', model_type, 'fold_' + str(commandline_arg_dic['split']))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save to json file
    json = json.dumps(commandline_arg_dic)
    print("Path of json file:", os.path.join(log_dir, "commandline.json"))
    f = open(os.path.join(log_dir, "commandline.json"), "w")
    f.write(json)
    f.close()


def memReport():
    import gc

    cnt_tensor = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size(), obj.is_cuda)
            cnt_tensor += 1
    print('Count: ', cnt_tensor)


def cpuStats():
    import sys
    import psutil

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
