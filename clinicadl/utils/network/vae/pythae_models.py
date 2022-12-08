from clinicadl.utils.network.vae.pythae_utils import build_encoder_decoder, Encoder, Decoder


def pythae_VAE(
    input_size=(1, 80, 96, 80),
    latent_space_size=128,
    feature_size=0,
    n_conv=3,
    io_layer_channels=32,
):
    from pythae.models import VAE, VAEConfig

    encoder_layers, mu_layer, logvar_layer, decoder_layers = build_encoder_decoder(
        input_size=input_size,
        latent_space_size=latent_space_size,
        feature_size=feature_size,
        n_conv=n_conv,
        io_layer_channels=io_layer_channels,
    )

    encoder = Encoder(encoder_layers, mu_layer, logvar_layer)
    decoder = Decoder(decoder_layers)

    vae_config = VAEConfig(
        input_dim=input_size,
        latent_dim=latent_space_size
    )
    return VAE(
        model_config=vae_config,
        encoder=encoder, # pass your encoder as argument when building the model
        decoder=decoder, # pass your decoder as argument when building the model
    )
