# coding: utf8

from . import cli
from clinicadl.tools.deep_learning import commandline_to_json


def main():

    parser = cli.parse_command_line()
    args = parser.parse_args()

    commandline = parser.parse_known_args()

    arguments = vars(args)

    if (arguments['task'] != 'preprocessing') \
            and (arguments['task'] != 'extract') \
            and (arguments['task'] != 'generate') \
            and (arguments['task'] != 'tsvtool'):
        commandline_to_json(commandline, arguments["mode_task"])

    args.func(args)

    
if __name__ == '__main__':
    main()
