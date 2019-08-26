from . import cli

def main():
    args = cli.parse_command_line()
    args.func(args)

if __name__ == '__main__':
    main()
