import click

patch_size = click.option(
    "-ps",
    "--patch_size",
    default=50,
    show_default=True,
    help="Patch size.",
)
stride_size = click.option(
    "-ss",
    "--stride_size",
    default=50,
    show_default=True,
    help="Stride size.",
)
