from . import cli
from clinicadl.tools.deep_learning import commandline_to_json

def main():

    parser = cli.parse_command_line()
    args = parser.parse_args()
    
    print(args)
    commandline = parser.parse_known_args()

    if hasattr(args, 'train_autoencoder'):
      model_type = 'autoencoder'
    else:
      model_type = 'cnn'

    arguments = vars(args)

    if (arguments['task'] != 'preprocessing') and (arguments['task'] != 'extract'):
      commandline_to_json(commandline, model_type)

    args.func(args)


if __name__ == '__main__':
    main()
