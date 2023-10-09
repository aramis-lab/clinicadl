from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_BetaVAE(BasePythae):
    def __init__(
        self,
        input_size,
        first_layer_channels,
        n_conv_encoder,
        feature_size,
        latent_space_size,
        n_conv_decoder,
        last_layer_channels,
        last_layer_conv,
        beta,
        gpu=False,
    ):

        from pythae.models import BetaVAE, BetaVAEConfig

        encoder, decoder = super(pythae_BetaVAE, self).__init__(
            input_size=input_size,
            first_layer_channels=first_layer_channels,
            n_conv_encoder=n_conv_encoder,
            feature_size=feature_size,
            latent_space_size=latent_space_size,
            n_conv_decoder=n_conv_decoder,
            last_layer_channels=last_layer_channels,
            last_layer_conv=last_layer_conv,
            gpu=gpu,
        )

        model_config = BetaVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            beta=beta
        )
        self.model = BetaVAE(
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
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
        )
