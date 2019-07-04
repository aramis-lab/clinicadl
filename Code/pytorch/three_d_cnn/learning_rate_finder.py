def lr_finder(model, dataloader, use_cuda, criterion, args):
    """
        Computes the balanced accuracy of the model

        :param model: the network (subclass of nn.Module)
        :param dataloader: a DataLoader wrapping a dataset
        :param use_cuda: if True a gpu is used
        :param full_return: if True also returns the sensitivities and specificities for a multiclass problem
        :return: balanced accuracy of the model (float)
        """
    import os
    import pandas as pd
    import numpy as np

    # Use tensors instead of arrays to avoid bottlenecks
    model.train()
    lr = args.learning_rate

    # Initialize outputs
    filename = os.path.join(args.output_dir, 'lr_multiplier-' + str(args.lr_multiplier) + '.tsv')
    results_df = pd.DataFrame(columns=['iteration', 'lr', 'total_loss'])
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs, labels = data['image'].cuda(), data['label'].cuda()
        else:
            inputs, labels = data['image'], data['label']

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()

        del inputs, outputs, labels

        if (i+1) % args.accumulation_steps == 0:
            optimizer = eval("torch.optim." + args.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)
            optimizer.step()
            model.zero_grad()
            # Write the total loss obtained for the learning rate
            row = np.array([i, lr, total_loss]).reshape(1, -1)
            row_df = pd.DataFrame(row, columns=['iteration', 'lr', 'total_loss'])
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')

            lr *= args.lr_multiplier
            total_loss = 0

            if lr > 1:
                break


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from utils.model import create_model
    from utils.data_utils import MRIDataset
    import os

    parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

    # Mandatory arguments
    parser.add_argument("diagnosis_tsv", type=str,
                        help="Path to tsv file of the population."
                             " To note, the column name should be participant_id, session_id and diagnosis.")
    parser.add_argument("output_dir", type=str,
                        help="Path to log dir for tensorboard usage.")
    parser.add_argument("img_dir", type=str,
                        help="Path to input dir of the MRI (preprocessed CAPS_dir).")
    parser.add_argument("model", type=str, choices=["Conv_3", "Conv_4", "Conv_5", "Test"],
                        help="model selected")

    # Data Management
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size for training. (default=1)")
    parser.add_argument('--accumulation_steps', '-asteps', default=5, type=int,
                        help='Accumulates gradients in order to increase the size of the batch')

    # Training arguments
    parser.add_argument("--learning_rate", "-lr", default=1e-8, type=float,
                        help="Learning rate of the optimization. (default=0.01)")
    parser.add_argument("--lr_multiplier", default=2, type=float,
                        help="The coefficient to increase the learning rate")

    # Optimizer arguments
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Uses gpu instead of cpu if cuda is available')

    args = parser.parse_args()
    args.transfer_learning = False

    model = create_model(args)
    dataset = MRIDataset(args.img_dir, args.diagnosis_tsv)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    lr_finder(model, dataloader, args.gpu, criterion, args)
