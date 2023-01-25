from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_WAE_MMD(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        kernel_choice,
        reg_weight,
        kernel_bandwidth,
        gpu=False,
    ):

        from pythae.models import WAE_MMD, WAE_MMD_Config

        encoder, decoder = super(pythae_WAE_MMD, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu,
            is_ae=True,
        )

        model_config = WAE_MMD_Config(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            kernel_choice=kernel_choice,
            reg_weight=reg_weight,
            kernel_bandwidth=kernel_bandwidth,
        )
        self.model = WAE_MMD(
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
            batch_size=batch_size,
        )
