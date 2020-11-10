# coding: utf8

from . import cli
import torch


def main():

    parser = cli.parse_command_line()
    args = parser.parse_args()

    if hasattr(args, 'use_cpu'):
        if not args.use_cpu and not torch.cuda.is_available():
            raise ValueError("No GPU is available. Please add the -cpu flag to run on CPU.")

    args.func(args)

    
if __name__ == '__main__':
    main()
