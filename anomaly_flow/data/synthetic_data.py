"""
    Atributos que serviram de Comporação para o Artigo Flow GAN: 

   ----------------------------------------------------
  | Feature Description |   Netflow Equivalent Name    |  
   ----------------------------------------------------
  | Flow Duration       |   FLOW_DURATION_MILLISECONDS |
  | Packets Sent        |   OUT_PKTS                   |
  | Bytes Per Packet    |   IN_BYTES / IN_PKTS         | 
   ----------------------------------------------------

"""
import pandas as pd
import tensorflow as tf

from anomaly_flow.train.constants import SEED

def generate_synthetic_data(generator, number_of_samples, columns, latent_vector_size, scaler):
    """
        Function to generate sythetic data usig the generated model.
    """
    z = tf.random.normal((number_of_samples, latent_vector_size), seed=SEED)
    synthetic_data = generator(z, training=False)
    unscaled_data = scaler.inverse_transform(synthetic_data.numpy())
    return pd.DataFrame(unscaled_data, columns=columns)
