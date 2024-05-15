"""
    NetflowV2 dataset, split to be used for anomaly detection.
"""
import os
import math
from copy import copy
from functools import partial
from typing import Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
import numpy as np
from anomaly_flow.data.dataset import AnomalyDetectionDataset
from anomaly_flow.data.dtypes_utils import dtypes_netflow
from anomaly_flow.utils.binary_processing import split_flag_columns

from anomaly_flow.train.constants import SEED


tf.random.set_seed(SEED)

class NetFlowV2(AnomalyDetectionDataset):
    """NetFlowV2 dataset, split to be used for anomaly detection.
    Note:
        The label 1 is for the ANOMALOUS class.
        The label 0 is for the NORMAL class.
    """

    def __init__(self, dataset_name, train_size = None):
        super().__init__()

        # Load Netflow Datasets
        self._dataset_name = dataset_name

        print(f"Loading dataset {self._dataset_name}...")

        df = pd.read_csv(
            f"{os.getcwd()}/datasets/{self._dataset_name}.csv.gz",
            dtype=dtypes_netflow
        )
        
        print(f"Loaded dataset {self._dataset_name} [OK]")

        # Pré processamento fazer one hot enconding de atributos descritivos
        print("Initialized Columns Split Preprocessing.")
        
        df = split_flag_columns(df)
        
        print("Finished Columns Split Preprocessing. [OK]")

        # Remove features não utilizadas para o treinamento
        df.drop(NetFlowV2.get_features_to_drop(), axis=1, inplace=True)

        # Pré processamento para remover valores muito grandes
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        threshold = np.finfo(np.float32).max
        df = df[df < threshold]

        df.dropna(inplace=True)

        X, y = df.drop(['Label'], axis=1), df['Label']
        self._columns = X.columns
        
        print('Experiments using the following columns: ')
        print(*list(self._columns), sep='\n')

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=SEED, shuffle=True)

        # Normalize the data between -1 and 1 (GAN output)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        if train_size is None:
            self.scaler.fit(X_train[y_train == 0])
            X_train = self.scaler.transform(X_train)
            y_train = y_train
        else:
            self.scaler.fit(X_train[y_train == 0])
            X_train = self.scaler.transform(X_train[:train_size])
            y_train = y_train[:train_size]

        self.validation_benign_samples = math.floor(0.1 * np.count_nonzero(y_train == 0))
        self.validation_attack_samples = math.floor(0.1 * np.count_nonzero(y_train == 1))

        X_test = self.scaler.transform(X_test)
        self.X_test = copy(X_test) 
        self.y_test = copy(y_test)

        # Cria datasets compatíveis com o treinamento dentro do tensorflow
        self._train_raw = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self._test_raw = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Binary Classification Approach
        self._num_classes = 2

        # Initializes the default Batch Size 
        self.batch_size = 128

        # Clear unused variables
        # del X_train
        # del y_train
        # del y_test
        # del df

    @staticmethod
    def get_features_to_drop() -> list: 
        """
            Atributos que enviesam o modelo e devem ser removidos. 
            IPV4_SRC_ADDR = Endereço de origem do fluxo de informação.
            IPV4_DST_ADDR = Endereço de de destino de fluxo de informação. 
            L7_PROTO = Protocolo da cama 7 de aplicação, específico para cada tipo de ataque DDoS. 
            L4_SRC_PORT = Porta de origem do fluxo de comunicação. 
            L4_DST_PORT = Porta de destino do fluxo de comunicação.
            FTP_COMMAND_RET_CODE = Código de retorno do comando FTP. 
            Attack = Label descritiva da classe do exemplo. 

            Demais atributos que possam prejudicar o desempenho devem ser adicionados nesta classe. 
        """
        __features_to_drop = [
            'Unnamed: 0',
            'IPV4_SRC_ADDR', 
            'IPV4_DST_ADDR', 
            'L7_PROTO', 
            'L4_SRC_PORT', 
            'L4_DST_PORT', 
            'FTP_COMMAND_RET_CODE',
            'Attack'
        ]

        return __features_to_drop

    def configure(
        self,
        batch_size: int,
        new_size: int,
        anomalous_label: Union[int, str, None] = None,
        shuffle_buffer_size: int = 100000,
        cache: bool = True,
        drop_remainder: bool = True,
        output_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Configure the dataset. This makes all the object properties valid (not None).
        Args:
            batch_size: The dataset batch size.
            new_size: (L) of de the input flow used to train the model.
            Verificar se esses pontos aqui precisam ser alterados
            anomalous_label: If the raw dataset contains label, all the elements with
                             "anomalous_label" are converted to element of
                             self.anomalous_label class.
            shuffle_buffer_size: Buffer size used during the tf.data.Dataset.shuffle call.
            cache: If True, cache the dataset.
            drop_remainder: If True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: A Tuple (min, max) containing the output range to use
                          for the processed images.
        """

        pipeline = partial(
            self.pipeline,
            new_size=new_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cache=cache,
            drop_remainder=drop_remainder,
            output_range=output_range,
        )

        self.batch_size = batch_size
        pipeline_train = partial(pipeline, is_training=True)
        pipeline_test = partial(pipeline, is_training=False)
        is_anomalous = lambda _, label: tf.equal(label, anomalous_label)
        is_normal = lambda _, label: tf.not_equal(label, anomalous_label)

        # Conclusão garantir que a validação contenha classes 0 e 1 !!!!!!
        per_class_dataset = [
            self._train_raw.filter(lambda _, y: tf.equal(y, label))
            for label in tf.range(self._num_classes, dtype=tf.float32)
        ]

        validation_raw = per_class_dataset[0].take(self.validation_benign_samples)
        train_raw = per_class_dataset[0].skip(self.validation_benign_samples)

        for i in range(1, self._num_classes):
            validation_raw = validation_raw.concatenate(per_class_dataset[i].take(self.validation_attack_samples))
            train_raw = train_raw.concatenate(per_class_dataset[i].skip(self.validation_attack_samples))

        # Train-data
        self._train_anomalous = (
            train_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_train)
        )

        self._train_normal = (
            train_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_train)
        )
        
        self._train = train_raw.map(
            lambda x, label: (
                x,
                tf.cast(tf.equal(label, anomalous_label), tf.float32),
            )
        ).apply(pipeline_train)

        # Validation data
        self._validation_anomalous = (
            validation_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._validation_normal = (
            validation_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )
        self._validation = validation_raw.map(
            lambda x, label: (
                x,
                tf.cast(tf.equal(label, anomalous_label), tf.float32),
            )
        ).apply(pipeline_test)

        # Test-data
        self._test_anomalous = (
            self._test_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._test_normal = (
            self._test_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )

        # Complete dataset with positive and negatives
        def _to_binary(x, y):
            if tf.equal(y, anomalous_label):
                return (x, self.anomalous_label)
            return (x, self.normal_label)

        self._test = self._test_raw.map(_to_binary).apply(pipeline_test)

        # Free memory
        del self._train_raw
        del self._test_raw
 
    def get_train_dataset(self):
        """
            Method that return the dataset used to train the model. 
        """
        return self.train_normal

    def get_test_dataset(self):
        """
            Method that return the dataset used to test the model. 
        """
        return self._test

    def get_train_size(self):
        """
            Method used to return the number of samples used to train the model. 
            Considering the batch size information.
            Note: The model uses only normal data during the train phase. 
        """
        return len(list(self._train_normal)) * self.batch_size

    def get_test_size(self):
        """
            Method used to return the number of samples used to train the model. 

            Note: The model uses only normal data during the train phase. 
        """
        return self._test.cardinality().numpy()
    
    def get_test_distribution(self):
        """
            Method to return the test distribution unchanged.
        """
        test_df = pd.DataFrame(self.scaler.inverse_transform(self.X_test), columns=self.get_columns())
        test_df["Label"] = self.y_test
        return test_df

    def get_columns(self) -> list():
        """
            Method to return the dataset columns loaded from the csv files.
        """
        return self._columns

    def get_name(self) -> str:
        """
            Method to return the dataset name. 
        """
        return self._dataset_name

    def get_scaler(self):
        """
            Method to result the scaler.
        """
        return self.scaler
