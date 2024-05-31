import click

# LR scheduler
adaptive_learning_rate = click.option(
    "--adaptive_learning_rate",
    "-alr",
    is_flag=True,
    help="Whether to diminish the learning rate",
)
