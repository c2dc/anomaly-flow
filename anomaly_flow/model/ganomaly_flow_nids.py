"""
    Module to define the models used to train the GANomaly model for Network Flows Classification.
"""
from tensorflow import keras as k
import anomaly_flow.train.constants as constants

KERNEL_INITIALIZER = k.initializers.RandomNormal(
    mean=0.0, stddev=0.02, seed=constants.SEED)
ALMOST_ONE = k.initializers.RandomNormal(mean=1.0, stddev=0.02, seed=constants.SEED)

# Represents the number of attributes of a sample
N_INPUTS = 52

class Decoder(k.Sequential):
    """
        Decoder Class Component.
    """
    def __init__(
        self,
        latent_space_dimension: int = 4,
        l2_penalty: float = 0.0,
    ):
        super().__init__(
            [
                k.layers.Dense(256, activation='relu', input_dim=latent_space_dimension,
                               kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2),
                k.layers.Dense(512, activation='relu',
                               kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2),
                k.layers.BatchNormalization(),
                k.layers.Dense(1024, activation='relu',
                               kernel_initializer=KERNEL_INITIALIZER),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Dense(
                    N_INPUTS,
                    activation='tanh',
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                    kernel_initializer=KERNEL_INITIALIZER
                )
            ]
        )


class Encoder(k.Sequential):
    """
        Class that represents the Encoder of the GAN architecture.
    """
    def __init__(
        self,
        latent_space_dimension: int = 100,
        l2_penalty: float = 0.0,
    ):
        super().__init__(
            [
                k.layers.Dense(1024, activation='relu', input_dim=N_INPUTS,
                               kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2),
                k.layers.Dense(512, activation='relu',
                               kernel_initializer=KERNEL_INITIALIZER),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Dense(256, activation='relu',
                               kernel_initializer=KERNEL_INITIALIZER),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Dense(
                    latent_space_dimension,
                    activation='relu',
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                    kernel_initializer=KERNEL_INITIALIZER
                )
            ]
        )


class Discriminator(k.Model):
    """
        Class that represents the Discriminator of the GAN architecture.
    """
    def __init__(
        self,
        l2_penalty: float = 0.0,
    ):

        super().__init__()

        self._backbone = k.Sequential(
            [
                k.layers.Dense(
                    1024, activation='relu', 
                    kernel_initializer=KERNEL_INITIALIZER, input_dim=N_INPUTS
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Dropout(0.2),
                k.layers.Dense(512, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2),
                k.layers.Dropout(0.2),
                k.layers.Dense(256, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2)
            ]
        )

        self._output = k.layers.Dense(
            1,kernel_initializer=KERNEL_INITIALIZER,
            use_bias=False,
            kernel_regularizer=k.regularizers.l2(l2_penalty),
            activation='sigmoid'
        )

    def call(self, inputs, training=True, mask=None):
        """
            Function used by tensorflow to fit and predict. 

            Overrides the default call to this specific architecture.
        """
        features = self._backbone(inputs, training=training, mask=mask)
        output = self._output(features)

        return output


def ganomaly_model(encoder, decoder, discriminator):
    """
        Function that defines the whole GAN implementation. 

        Encoder -> Decoder -> Discriminator

    """
    model = k.Sequential()
    model.add(encoder)
    model.add(decoder)
    model.add(discriminator)
    return model
