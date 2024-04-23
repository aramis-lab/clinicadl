import click

from clinicadl.generate.generate_config import GenerateHypometabolicConfig

config_hypometabolic = GenerateHypometabolicConfig.model_fields
pathology = click.option(
    "--pathology",
    "-p",
    type=config_hypometabolic["pathology"].annotation,
    default=config_hypometabolic["pathology"].default,
    help="Pathology applied. To chose in the following list: [ad, bvftd, lvppa, nfvppa, pca, svppa]",
)
anomaly_degree = click.option(
    "--anomaly_degree",
    "-anod",
    type=config_hypometabolic["anomaly_degree"].annotation,
    default=config_hypometabolic["anomaly_degree"].default,
    help="Degrees of hypo-metabolism applied (in percent)",
)
sigma = click.option(
    "--sigma",
    type=config_hypometabolic["sigma"].annotation,
    default=config_hypometabolic["sigma"].default,
    help="It is the parameter of the gaussian filter used for smoothing.",
)
