import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os, shutil
from skimage.transform import resize
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

def train(model, train_loader, use_cuda, criterion, optimizer, writer_train, epoch_i):
    """
    This is the function to train the model
    :param model:
    :param train_loader:
    :param use_cuda:
    :param criterion:
    :param optimizer:
    :param writer:
    :param epoch_i:
    :return:
    """
    # main training loop
    correct_cnt = 0.0
    model.train() ## set the module to training mode

    for i, train_data in enumerate(train_loader):
        # for each it, the train data contains batch_size * n_slices_in_each_subject images
        loss_batch = 0.0
        acc_batch = 0.0
        for j in range(len(train_data)):
            data_dic = train_data[j]
            if use_cuda:
                imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda()
            else:
                imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])
            integer_encoded = labels.data.cpu().numpy()
            # target should be LongTensor in loss function
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
            print 'The group true label is %s' % str(labels)
            if use_cuda:
                ground_truth = ground_truth.cuda()
            train_output = model(imgs)
            _, predict = train_output.topk(1)
            loss = criterion(train_output, ground_truth)
            loss_batch += loss
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            correct_cnt += correct_this_batch
            accuracy = float(correct_this_batch) / len(ground_truth)
            acc_batch += accuracy
            print ("For batch %d slice %d training loss is : %f") % (i, j, loss.item())
            print ("For batch %d slice %d training accuracy is : %f") % (i, j, accuracy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer_train.add_scalar('training_accuracy', acc_batch / len(train_data), i + epoch_i * len(train_loader.dataset))
        writer_train.add_scalar('training_loss', loss_batch / len(train_data), i + epoch_i * len(train_loader.dataset))
        ## add image
        writer_train.add_image('example_image', imgs.int(), i + epoch_i * len(train_loader.dataset))


    return imgs

def validate(model, valid_loader, use_cuda, criterion, writer_valid, epoch_i):
    """
    This is the function to validate the CNN with validation data
    :param model:
    :param valid_loader:
    :param use_cuda:
    :param criterion:
    :param writer:
    :param epoch_i:
    :return:
    """
    correct_cnt = 0
    acc = 0.0
    model.eval()
    for i, valid_data in enumerate(valid_loader):
        loss_batch = 0.0
        acc_batch = 0.0
        for j in range(len(valid_data)):
            data_dic = valid_data[j]
            if use_cuda:
                imgs, labels = Variable(data_dic['image'], volatile=True).cuda(), Variable(data_dic['label'],
                                                                                           volatile=True).cuda()
            else:
                imgs, labels = Variable(data_dic['image'], volatile=True), Variable(data_dic['label'],
                                                                                    volatile=True)
            integer_encoded = labels.data.cpu().numpy()
            # target should be LongTensor in loss function
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
            print 'The group true label is %s' % str(labels)
            if use_cuda:
                ground_truth = ground_truth.cuda()
            valid_output = model(imgs)
            _, predict = valid_output.topk(1)
            loss = criterion(valid_output, ground_truth)
            loss_batch += loss
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            correct_cnt += correct_this_batch
            accuracy = float(correct_this_batch) / len(ground_truth)
            acc_batch += accuracy

            print ("For batch %d slice %d validation loss is : %f") % (i, j, loss.item())
            print ("For batch %d slice %d validation accuracy is : %f") % (i, j, accuracy)

        writer_valid.add_scalar('validation_accuracy', acc_batch / len(valid_data), i + epoch_i * len(valid_loader.dataset))
        writer_valid.add_scalar('validation_loss', loss_batch / len(valid_data), i + epoch_i * len(valid_loader.dataset))

        acc += acc_batch / len(valid_data)

    acc_mean = acc / len(valid_loader)

    return acc_mean

def test(model, test_loader, use_cuda, writer_test):
    """
    This is the function to test the CNN with testing data
    :param model:
    :param test_loader:
    :param use_cuda:
    :param criterion:
    :param writer:
    :return:
    """
    correct_cnt = 0
    model.eval()
    for i, test_data in enumerate(test_loader):
        vote = []
        acc_batch = 0.0

        for j in range(len(test_data)):
            data_dic = test_data[j]
            if use_cuda:
                imgs, labels = Variable(data_dic['image'], volatile=True).cuda(), Variable(data_dic['label'],
                                                                                           volatile=True).cuda()
            else:
                imgs, labels = Variable(data_dic['image'], volatile=True), Variable(data_dic['label'],
                                                                                    volatile=True)

            ## fpr slice-level accuracy
            integer_encoded = labels.data.cpu().numpy()
            # target should be LongTensor in loss function
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
            print 'The group true label is %s' % str(labels)
            if use_cuda:
                ground_truth = ground_truth.cuda()
            test_output = model(imgs)
            _, predict = test_output.topk(1)
            vote.append(predict)
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            accuracy_slice = float(correct_this_batch) / len(ground_truth)
            acc_batch += accuracy_slice

            print ("For batch %d slice %d test accuracy is : %f") % (i, j, accuracy_slice)

        writer_test.add_scalar('test_accuracy_slice', acc_batch / len(test_data), i)

        ## for subject-level accuracy
        vote = torch.cat(vote, 1)
        final_vote, _ = torch.mode(vote, 1) ## This is the majority vote for each subject, based on all the slice-level results
        ground_truth = test_data[0]['label']
        correct_this_batch_subject = (final_vote.cpu().data == ground_truth).sum()
        accuracy_subject = float(correct_this_batch_subject) / len(ground_truth)

        print("Subject level for batch %d testing accuracy is : %f") % (i, accuracy_subject)
        writer_test.add_scalar('test_accuracy_subject', accuracy_subject, i)

    return accuracy_subject

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    import shutil, os

    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(checkpoint_dir, 'model_best.pth.tar'))

def split_subjects_to_tsv(diagnoses_tsv, n_splits=5, test_size=0.2, fold=0):

    df = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')
    if 'diagnosis' not in list(df.columns.values):
        raise Exception('Diagnoses file is not in the correct format.')
    diagnoses_list = list(df.diagnosis)
    unique = list(set(diagnoses_list))
    y = np.array([unique.index(x) for x in diagnoses_list]) ### Here, AD is 0 and CN is 1

    splits = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    n_iteration = 0
    for train_index, test_index in splits.split(np.zeros(len(y)), y):

        # for training
        df_train = df.iloc[train_index]
        df_test_valid = df.iloc[test_index]
        y_test_valid = y[test_index]

        ### split the test data into validation and test
        skf_2 = StratifiedShuffleSplit(n_splits=2, test_size=0.5)
        for test_ind, valid_ind in skf_2.split(df_test_valid, y_test_valid):
            print("SPLIT iteration:", "Test:", test_ind, "Validation", valid_ind)

        df_test = df_test_valid.iloc[test_ind]
        df_valid = df_test_valid.iloc[valid_ind]

        df_train.to_csv(path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(n_iteration) + '_train.tsv'), sep='\t', index=False)
        df_test.to_csv(path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(n_iteration) + '_test.tsv'), sep='\t', index=False)
        df_valid.to_csv(path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(n_iteration) + '_valid.tsv'), sep='\t', index=False)
        n_iteration += 1

    training_tsv = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_train.tsv')
    test_tsv = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_train.tsv')
    valid_tsv = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_test.tsv')

    return training_tsv, test_tsv, valid_tsv

def check_and_clean(d):

  if os.path.exists(d):
      shutil.rmtree(d)
  os.mkdir(d)

class AD_Standard_2DSlicesData(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, root_dir, data_file, transform=None, slice=slice):
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.

        for each view:
            AX_SCETION = "[:, :, slice_i]"
            COR_SCETION = "[:, slice_i, :]"
            SAG_SCETION = "[slice_i, :, :]"

        """
        self.root_dir = root_dir
        self.data_file = data_file
        self.transform = transform

    def __len__(self):
        df = pd.io.parsers.read_csv(self.data_file, sep='\t')
        if ('diagnosis' != list(df.columns.values)[2]) and ('session_id' != list(df.columns.values)[1]) and (
            'participant_id' != list(df.columns.values)[0]):
            raise Exception('the data file is not in the correct format.')
        img_list = list(df['participant_id'])
        return len(img_list)

    def __getitem__(self, idx):
        df = pd.io.parsers.read_csv(self.data_file, sep='\t')
        if ('diagnosis' != list(df.columns.values)[2]) and ('session_id' != list(df.columns.values)[1]) and (
            'participant_id' != list(df.columns.values)[0]):
            raise Exception('the data file is not in the correct format.')
        img_list = list(df['participant_id'])
        sess_list = list(df['session_id'])
        label_list = list(df['diagnosis'])

        img_name = img_list[idx]
        img_label = label_list[idx]
        sess_name = sess_list[idx]
        image_path = os.path.join(self.root_dir, img_name, sess_name, 'anat',
                                  img_name + '_' + sess_name + '_T1w.nii.gz')
        image = nib.load(image_path)
        samples = []
        if img_label == 'CN':
            label = 0
        elif img_label == 'AD':
            label = 1
        elif img_label == 'MCI':
            label = 2

        AXimageList, n_image_ax = axKeySlice(image)
        CORimageList, n_image_cor = corKeySlice(image)
        SAGimageList, n_image_sag = sagKeySlice(image)

        for img2DList in (AXimageList, CORimageList, SAGimageList):
            for image2D in img2DList:
                if self.transform:
                    image2D = self.transform(image2D)
                sample = {'image': image2D, 'label': label}
                samples.append(sample)
        random.shuffle(samples)
        return samples


def getSlice(image_array, view):
    """

    :param image_array:
    :param keyIndex:
    :param view:
    :return:
    """

    slice_2Dimgs = []
    slice_list = range(15, image_array.shape[view] - 15) # delete the first 15 slice and last 15 slice

    scalar = MinMaxScaler(feature_range=(0, 255), copy=False)
    for i in slice_list:
        ## sagital
        if view == 0:
            slice_select_0 = scalar.fit_transform(np.resize(image_array[i - 1, :, :], [image_array[:, :, i].shape[0],
                                                                                       image_array[:, :, i].shape[
                                                                                           1]])).astype('uint8')
            slice_select_1 = scalar.fit_transform(
                np.resize(image_array[i, :, :], [image_array[:, :, i].shape[0], image_array[:, :, i].shape[1]])).astype(
                'uint8')
            slice_select_2 = scalar.fit_transform(np.resize(image_array[i + 1, :, :], [image_array[:, :, i].shape[0],
                                                                                       image_array[:, :, i].shape[
                                                                                           1]])).astype('uint8')
        ## coronal
        if view == 1:
            slice_select_0 = scalar.fit_transform(np.resize(image_array[:, i - 1, :], [image_array[:, :, i].shape[0],
                                                                                       image_array[:, :, i].shape[
                                                                                           1]])).astype('uint8')
            slice_select_1 = scalar.fit_transform(
                np.resize(image_array[:, i, :], [image_array[:, :, i].shape[0], image_array[:, :, i].shape[1]])).astype(
                'uint8')
            slice_select_2 = scalar.fit_transform(np.resize(image_array[:, i + 1, :], [image_array[:, :, i].shape[0],
                                                                                       image_array[:, :, i].shape[
                                                                                           1]])).astype('uint8')
        ## axial
        if view == 2:
            slice_select_0 = scalar.fit_transform(np.resize(image_array[:, :, i - 1], [image_array[:, :, i].shape[0],
                                                                                       image_array[:, :, i].shape[
                                                                                           1]])).astype('uint8')
            slice_select_1 = scalar.fit_transform(
                np.resize(image_array[:, :, i], [image_array[:, :, i].shape[0], image_array[:, :, i].shape[1]])).astype(
                'uint8')
            slice_select_2 = scalar.fit_transform(np.resize(image_array[:, :, i + 1], [image_array[:, :, i].shape[0],
                                                                                       image_array[:, :, i].shape[
                                                                                           1]])).astype('uint8')

        slice_2Dimg = np.stack((slice_select_0, slice_select_1, slice_select_2), axis=2)
        if len(slice_2Dimg.shape) > 3 and slice_2Dimg.shape[3] == 1:
            slice_2Dimg_resize = np.resize(slice_2Dimg,
                                           (slice_2Dimg.shape[0], slice_2Dimg.shape[1], slice_2Dimg.shape[2]))
            slice_2Dimgs.append(slice_2Dimg_resize)
        else:
            slice_2Dimgs.append(slice_2Dimg)

    return slice_2Dimgs, len(slice_list)


def axKeySlice(image):
    image_array = np.array(image.get_data())
    images, n_image = getSlice(image_array, 2)
    return images, n_image


def corKeySlice(image):
    image_array = np.array(image.get_data())
    images, n_image = getSlice(image_array, 1)
    return images, n_image


def sagKeySlice(image):
    image_array = np.array(image.get_data())
    images, n_image = getSlice(image_array, 0)
    return images, n_image

class CustomResize(object):
    def __init__(self, trg_size):
        self.trg_size = trg_size

    def __call__(self, img):
        resized_img = self.resize_image(img, self.trg_size)
        return resized_img

    def resize_image(self, img_array, trg_size):
        res = resize(img_array, trg_size, mode='reflect', preserve_range=True, anti_aliasing=False)

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
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            # backward compatibility
            return img.float()
