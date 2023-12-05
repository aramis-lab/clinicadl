from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_VAE(BasePythae):
    def __init__(
        self,
        input_size,
        first_layer_channels,
        n_block_encoder,
        feature_size,
        latent_space_size,
        n_block_decoder,
        last_layer_channels,
        last_layer_conv,
        n_layer_per_block_encoder,
        n_layer_per_block_decoder,
        block_type,
        gpu=False,
    ):
        from pythae.models import VAE, VAEConfig

        encoder, decoder = super(pythae_VAE, self).__init__(
            input_size=input_size,
            first_layer_channels=first_layer_channels,
            n_block_encoder=n_block_encoder,
            feature_size=feature_size,
            latent_space_size=latent_space_size,
            n_block_decoder=n_block_decoder,
            last_layer_channels=last_layer_channels,
            last_layer_conv=last_layer_conv,
            n_layer_per_block_encoder=n_layer_per_block_encoder,
            n_layer_per_block_decoder=n_layer_per_block_decoder,
            block_type=block_type,
            gpu=gpu,
        )

        model_config = VAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
        )
        self.model = VAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size, optimizer):
        from pythae.trainers import BaseTrainerConfig

        return BaseTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            optimizer_cls=optimizer,
            # amp=True,
        )
