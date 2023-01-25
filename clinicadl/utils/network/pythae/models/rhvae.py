from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_RHVAE(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        n_lf,
        eps_lf,
        beta_zero,
        temperature,
        regularization,
        gpu=False,
    ):

        from pythae.models import RHVAE, RHVAEConfig

        encoder, decoder = super(pythae_RHVAE, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu
        )

        model_config = RHVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            temperature=temperature,
            regularization=regularization,
        )
        self.model = RHVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size):
        from pythae.trainers import BaseTrainerConfig
        return BaseTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
