# coding: utf8

from . import cli
from clinicadl.tools.deep_learning import commandline_to_json, write_requirements_version
import torch


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
        write_requirements_version(args.output_dir)

    if hasattr(args, "use_cpu"):
        if not args.use_cpu and not torch.cuda.is_available():
            raise ValueError("No GPU is available. Please add the -cpu flag to run on CPU.")

    args.func(args)

    
if __name__ == '__main__':
    main()
