# coding: utf8

from . import cli
from clinicadl.tools.deep_learning import commandline_to_json


def main():

    parser = cli.parse_command_line()
    args = parser.parse_args()

    commandline = parser.parse_known_args()

    if hasattr(args, 'train_autoencoder'):
        task_type = 'autoencoder'
    else:
        task_type = 'cnn'

    arguments = vars(args)

    if (arguments['task'] != 'preprocessing') \
            and (arguments['task'] != 'extract') \
            and (arguments['task'] != 'generate') \
            and (arguments['task'] != 'tsvtool'):
        commandline_to_json(commandline, task_type)

    args.func(args)

    
if __name__ == '__main__':
    main()
