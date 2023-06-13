from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_AE(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        gpu=False,
    ):

        from pythae.models import AE, AEConfig

        encoder, decoder = super(pythae_AE, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu,
            is_ae=True,
        )

        model_config = AEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
        )
        self.model = AE(
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
