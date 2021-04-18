# coding: utf8

import cli
import torch


def main():

    parser = cli.parse_command_line()
    args = parser.parse_args()

    if (args.version):
        import clinicaaddl
        print(f"ClinicaDL version is: {clinicaaddl.__version__}")
        exit(0)
    if hasattr(args, 'use_cpu'):
        if not args.use_cpu and not torch.cuda.is_available():
            raise ValueError("No GPU is available. Please add the -cpu flag to run on CPU.")

    if not args.task:
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
