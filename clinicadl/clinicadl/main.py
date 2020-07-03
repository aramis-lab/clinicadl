# coding: utf8

from . import cli
from clinicadl.tools.deep_learning import commandline_to_json
import torch
from os import path
import sys


def main():

    parser = cli.parse_command_line()
    args = parser.parse_args()

    if args.task == "train" and args.mode == "slice":
        args.mode_task = "cnn"

    commandline = parser.parse_known_args()

    arguments = vars(args)

    if (arguments['task'] != 'preprocessing') \
            and (arguments['task'] != 'extract') \
            and (arguments['task'] != 'generate') \
            and (arguments['task'] != 'tsvtool') \
            and (arguments['task'] != 'quality_check') \
            and (arguments['task'] != 'classify'):
        commandline_to_json(commandline)
        text_file = open(path.join(args.output_dir, 'environment.txt'), 'w')
        text_file.write('Version of python: %s \n' % sys.version)
        text_file.write('Version of pytorch: %s \n' % torch.__version__)
        text_file.close()

    if arguments['task'] in ['train', 'quality_check']:
        if not args.use_cpu and not torch.cuda.is_available():
            raise ValueError("No GPU is available. Please add the -cpu flag to run on CPU.")

    args.func(args)

    
if __name__ == '__main__':
    main()
