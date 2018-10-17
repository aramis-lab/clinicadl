import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import shutil
import warnings


def train(model, train_loader, valid_loader, criterion, optimizer, fold, options):
    """
    This is the function to train the model
    :param model:
    :param train_loader:
    :param valid_loader:
    :param criterion:
    :param optimizer:
    :param fold:
    :param options:
    :return:
    """
    # Create writers
    from tensorboardX import SummaryWriter
    writer_train = SummaryWriter(log_dir=(os.path.join(options.log_dir, "fold" + str(fold), "train")))

    # Initialize counters
    best_valid_accuracy = 0.0
    model.train()  # set the module to training mode

    for epoch in range(options.epochs):
        total_correct_cnt = 0.0
        print("At %d-th epoch." % epoch)

        model.zero_grad()
        evaluation_flag = True
        for i, data in enumerate(train_loader):

            if options.gpu:
                imgs, labels = Variable(data['image']).cuda(), Variable(data['label']).cuda()
            else:
                imgs, labels = Variable(data['image']), Variable(data['label'])

            train_output = model(imgs)
            _, predict = train_output.topk(1)
            loss = criterion(train_output, labels)
            batch_correct_cnt = (predict.squeeze(1) == labels).sum().float()
            total_correct_cnt += batch_correct_cnt
            accuracy = float(batch_correct_cnt) / len(labels)
            loss.backward()

            if (i+1) % options.accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

                # Evaluate the model only when no gradients are accumulated
                if(i+1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)
                    acc_mean_valid = test(model, valid_loader, options.gpu)
                    print("Scan level validation accuracy is %f at the end of iteration %d" % (acc_mean_valid, i))

                    is_best = acc_mean_valid > best_valid_accuracy
                    best_valid_accuracy = max(acc_mean_valid, best_valid_accuracy)
                    save_checkpoint({'model': model.state_dict(),
                                     'epoch': epoch,
                                     'valid_acc': acc_mean_valid},
                                    is_best,
                                    os.path.join(options.log_dir, "fold" + str(fold)))

        # If no evaluation has been performed, evaluate once at the end of the epoch
        if evaluation_flag:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset')
            acc_mean_valid = test(model, valid_loader, options.gpu)
            print("Scan level validation accuracy is %f at the end of epoch %d" % (acc_mean_valid, i))

            is_best = acc_mean_valid > best_valid_accuracy
            best_valid_accuracy = max(acc_mean_valid, best_valid_accuracy)
            save_checkpoint({'model': model.state_dict(),
                             'epoch': epoch,
                             'valid_acc': acc_mean_valid},
                            is_best,
                            os.path.join(options.log_dir, "log_dir" + "fold" + str(fold)))

            writer_train.add_scalar('training_accuracy', accuracy / len(data), i + epoch * len(train_loader.dataset))
            writer_train.add_scalar('training_loss', loss / len(data), i + epoch * len(train_loader.dataset))

        print('Total correct labels: %d / %d' % (total_correct_cnt, len(train_loader) * train_loader.batch_size))
        # at then end of each epoch, we validate one time for the model with the validation data


# def test(model, valid_loader, use_cuda, criterion, writer_valid, epoch_i):
#     """
#     This is the function to validate the CNN with validation data
#     :param model:
#     :param valid_loader:
#     :param use_cuda:
#     :param criterion:
#     :param writer_valid:
#     :param epoch_i:
#     :return:
#     """
#     acc = 0.0
#     model.eval()
#     for i, data in enumerate(valid_loader):
#         if use_cuda:
#             imgs, labels = Variable(data['image'], volatile=True).cuda(), Variable().cuda()
#         else:
#             imgs, labels = Variable(data['image'], volatile=True), Variable(data['label'],
#                                                                                     volatile=True)
#         # integer_encoded = labels.data.cpu().numpy()
#         # # target should be LongTensor in loss function
#         # ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
#         print('The group true label is %s' % str(labels))
#         # if use_cuda:
#         #     ground_truth = ground_truth.cuda()
#         valid_output = model(imgs)
#         _, predict = valid_output.topk(1)
#         loss = criterion(valid_output, labels)
#         correct_this_batch = (predict.squeeze(1) == labels).sum().float()
#         accuracy = float(correct_this_batch) / len(labels)
#
#         print("For batch %d validation loss is : %f") % (i, loss.item())
#         print("For batch %d validation accuracy is : %f") % (i, accuracy)
#
#         writer_valid.add_scalar('validation_accuracy', accuracy / len(data), i + epoch_i * len(valid_loader.dataset))
#         writer_valid.add_scalar('validation_loss', loss / len(data), i + epoch_i * len(valid_loader.dataset))
#
#         acc += accuracy / len(data)
#
#     acc_mean = acc / len(valid_loader)
#     print('Mean accuracy: %f' % acc_mean)
#
#     return acc_mean


def test(model, dataloader, use_cuda, verbose=False, full_return=False):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a dataloader wrapping a dataset
    :param gpu: if True a gpu is used
    :param full_return: if True also returns the sensitivities and specificities for a multiclass problem
    :return: balanced accuracy of the model (float)
    """
    model.eval()

    predicted_list = []
    truth_list = []
    model = model.eval()

    for i, data in enumerate(dataloader):
        if use_cuda:
            inputs, labels = data['image'].cuda(), data['label'].cuda()
        else:
            inputs, labels = data['image'], data['label']

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted_list = predicted_list + predicted.tolist()
        truth_list = truth_list + labels.tolist()

    # Computation of the balanced accuracy
    component = len(np.unique(truth_list))

    cluster_diagnosis_prop = np.zeros(shape=(component, component))
    for i, predicted in enumerate(predicted_list):
        truth = truth_list[i]
        cluster_diagnosis_prop[predicted, truth] += 1

    acc = 0
    sensitivity = np.zeros(component)
    specificity = np.zeros(component)
    for i in range(component):
        diag_represented = np.argmax(cluster_diagnosis_prop[i])
        acc += cluster_diagnosis_prop[i, diag_represented] / np.sum(cluster_diagnosis_prop.T[diag_represented])

        # Computation of sensitivity
        sen_array = cluster_diagnosis_prop[i]
        if np.sum(sen_array) == 0:
            sensitivity[diag_represented] = None
        else:
            sensitivity[diag_represented] = sen_array[diag_represented] / np.sum(sen_array) * 100

        # Computation of specificity
        spe_array = np.delete(cluster_diagnosis_prop, i, 0)
        if np.sum(spe_array) == 0:
            specificity[diag_represented] = None
        else:
            specificity[diag_represented] = (1 - np.sum(spe_array[:, diag_represented]) / np.sum(spe_array)) * 100

    acc = acc * 100 / component
    if verbose:
        print('Accuracy of diagnosis: ' + str(acc))
        print('Sensitivity of diagnoses:', sensitivity)
        print('Specificity of diagnoses:', specificity)

    if full_return:
        return acc, sensitivity, specificity

    return acc


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def load_best(model, checkpoint_dir):
    from copy import deepcopy

    best_model = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, 'model_best.pth.tar'))
    best_model.load_state_dict(param_dict['model'])
    return best_model, param_dict['epoch']


def check_and_clean(d):

    if os.path.exists(d):
        shutil.rmtree(d)
    os.mkdir(d)
