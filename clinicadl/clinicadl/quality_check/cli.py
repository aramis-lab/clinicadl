import click


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(name="quality-check")
def cli() -> None:
    """Quality check of clinica t1-linear and t1-volume pipeline's outputs"""
    pass


cli.add_command()

if __name__ == "__main__":
    cli()
