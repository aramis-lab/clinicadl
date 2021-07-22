import click


@click.group(name="interpret")
def cli() -> None:
    """Description"""
    pass


if __name__ == "__main__":
    cli()
