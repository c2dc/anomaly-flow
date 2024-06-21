# This file was modified to meet the necessary requirements of the Anomaly-Flow project,
# which aims to apply the GANomaly model to detect DDoS attacks using network flows. 
#
# Modifications/Additions to this file are Copyright (c) Leonardo Melo 2024
# and are licensed under the MIT license. 
# You may obtain the full License description at 
# 
# https://github.com/c2dc/anomaly-flow/blob/main/LICENSE 
# 
# When reusing the source code, follow the presented license notice and the original notice as shown below:
#
# Copyright 2021 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer for the GANomaly model."""

from typing import Dict, Set, Union
from pathlib import Path

import json
import copy
import numpy as np
import tensorflow as tf

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.losses.ganomaly import AdversarialLoss, generator_bce
from anomaly_toolbox.trainers.trainer import Trainer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from tensorflow import keras as k

from anomaly_flow.model.ganomaly_flow_nids import Decoder, Discriminator, Encoder, ganomaly_model
from anomaly_flow.view.roc_plot import plot_roc_curve
from anomaly_flow.view.distribution_plot import generate_anomaly_plot
import anomaly_flow.train.constants as constants

class GANomaly(Trainer):
    """GANomaly Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
        log_dir: Path,
    ):
        """Initialize GANomaly Networks."""
        super().__init__(
            dataset=dataset, hps=hps, summary_writer=summary_writer, log_dir=log_dir
        )

        # Save dataset name
        self.dataset_name = dataset.get_name()

        # Discriminator
        self.discriminator = Discriminator(l2_penalty=0.2)

        # Generator (aka Decoder)
        self.generator = Decoder(
            latent_space_dimension=self._hps["latent_vector_size"],
            l2_penalty=0.2,
        )

        # Encoder
        self.encoder = Encoder(
            latent_space_dimension=self._hps["latent_vector_size"],
            l2_penalty=0.2,
        )

        fake_batch_size = (1, 52)
        self.discriminator(tf.zeros(fake_batch_size))
        self.discriminator.summary()

        self.encoder(tf.zeros(fake_batch_size))
        self.encoder.summary()

        fake_latent_vector = (1, self._hps["latent_vector_size"])
        self.generator(tf.zeros(fake_latent_vector))
        self.generator.summary()

        # Losses
        self._mse = k.losses.MeanSquaredError()
        self._mae = k.losses.MeanAbsoluteError()

        self.optimizer_ge = k.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )
        self.optimizer_d = k.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )

        # Training Metrics
        self.epoch_g_loss_avg = k.metrics.Mean(name="epoch_generator_loss")
        self.epoch_d_loss_avg = k.metrics.Mean(name="epoch_discriminator_loss")
        self.epoch_e_loss_avg = k.metrics.Mean(name="epoch_encoder_loss")

        self._auc_rc = k.metrics.AUC(
            name="auc_rc", curve="PR", num_thresholds=100)
        self._auc_roc = k.metrics.AUC(
            name="auc_roc", curve="ROC", num_thresholds=100)

        self._f1_value = 0.0
        self._f2_value = 0.0
        self._acc_value = 0.0
        self.anomaly_threshold = 0.5

        self.keras_metrics = {
            metric.name: metric
            for metric in [
                self.epoch_d_loss_avg,
                self.epoch_g_loss_avg,
                self.epoch_e_loss_avg,
                self._auc_rc,
                self._auc_roc,
            ]
        }

        self._minmax = AdversarialLoss(from_logits=False)

        # Flatten op
        self._flatten = k.layers.Flatten()

    def get_model(self):
        """
            Method to retrieve the complete GAN model.
        """
        return ganomaly_model(self.encoder, self.generator, self.discriminator)

    def get_generator(self):
        """
            Method used to get the Generator Model 
        """
        return self.generator

    @staticmethod
    def hyperparameters() -> Set[str]:
        """List of the hyperparameters name used by the trainer."""
        return {
            "learning_rate",
            "latent_vector_size",
            "adversarial_loss_weight",
            "contextual_loss_weight",
            "enc_loss_weight",
        }

    def train(
        self,
        epochs: int,
        adversarial_loss_weight: float, 
        contextual_loss_weight: float,
        enc_loss_weight: float,
        step_log_frequency: int = 100,
    ):
        """
            Method used to train the GAN model.
        """
        best_auc_rc, best_auc_roc = -1, -1
        
        for epoch in range(epochs):
            # A variavel train normal está sendo utilizada, tenho que preencher ela lá no dataset
            for batch in self._dataset.train_normal:
                x, _ = batch
                x_numpy = x.numpy()
                x = tf.constant(x)
                # Perform the train step
                g_z, g_ex, d_loss, g_loss, e_loss = self.train_step(
                    x,
                    adversarial_loss_weight,
                    contextual_loss_weight,
                    enc_loss_weight,
                )

                # Update the losses metrics
                self.epoch_g_loss_avg.update_state(g_loss)
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_e_loss_avg.update_state(e_loss)

                step = self.optimizer_d.iterations.numpy()
                learning_rate = self.optimizer_ge.learning_rate.numpy()

                if tf.equal(tf.math.mod(step, step_log_frequency), 0):
                    with self._summary_writer.as_default():
                        tf.summary.scalar(
                            "learning_rate", learning_rate, step=step)
                        tf.summary.scalar(
                            "d_loss",
                            self.epoch_d_loss_avg.result(),
                            step=step,
                        )
                        tf.summary.scalar(
                            "g_loss",
                            self.epoch_g_loss_avg.result(),
                            step=step,
                        )
                        tf.summary.scalar(
                            "e_loss",
                            self.epoch_e_loss_avg.result(),
                            step=step,
                        )

                    tf.print(
                        "Step ",
                        step,
                        ". d_loss: ",
                        self.epoch_d_loss_avg.result(),
                        ", g_loss: ",
                        self.epoch_g_loss_avg.result(),
                        ", e_loss: ",
                        self.epoch_e_loss_avg.result(),
                        "lr: ",
                        learning_rate,
                    )

            # Epoch end
            tf.print(epoch, "Epoch completed")

            # Model selection
            self._auc_rc.reset_state()
            self._auc_roc.reset_state()
            self._f1_value = 0.0
            self._f2_value = 0.0 
            self._acc_value = 0.0 

            group_labels = list()
            group_scores = list()

            for batch in self._dataset.validation:
                x, labels_test = batch
                x_numpy = x.numpy()

                anomaly_scores = self._compute_anomaly_scores(
                    x, self.encoder, self.generator
                )

                group_labels.extend(labels_test)
                group_scores.extend(anomaly_scores)
          
                self._auc_rc.update_state(labels_test, anomaly_scores)
                self._auc_roc.update_state(labels_test, anomaly_scores)

            
            # self._compute_anomaly_score_threshold(group_scores)
            fpr, tpr, thresholds = roc_curve(group_labels, group_scores)
            self.anomaly_threshold = thresholds[self._compute_youden_position(tpr, fpr)]

            expected, anomaly = self._get_classified_data(group_labels, group_scores)
            self._f1_value = self._compute_f1_score(expected, anomaly)
            self._f2_value = self._compute_f2_score(expected, anomaly)
            self._acc_value = self._comput_accuracy(expected, anomaly)

            print(f"{self.dataset_name}: ")
            print(f"Validation f1-score: {self._f1_value}")
            print(f"Validation f2-score: {self._f2_value}")
            print(f"Validation accuracy: {self._acc_value}")

            # Save the model when AUC-RC is the best
            current_auc_rc = self._auc_rc.result()
            print(f"Validation Current AUC RC: {current_auc_rc}")

            if best_auc_rc < current_auc_rc:
                tf.print("Best AUC-RC on validation set: ", current_auc_rc)

                # Replace the best
                best_auc_rc = current_auc_rc

                base_path = self._log_dir / self.dataset_name / "results" / "auc_rc"
                self.generator.save(
                    str(base_path / "generator"), overwrite=True)
                self.encoder.save(str(base_path / "encoder"), overwrite=True)
                self.discriminator.save(
                    str(base_path / "discriminator"), overwrite=True
                )

                with open(base_path / "validation.json", "w", encoding='utf-8') as fp:
                    json.dump(
                        {
                            "value": float(best_auc_rc),
                        },
                        fp,
                    )
            # Save the model when AUC-ROC is the best
            current_auc_roc = self._auc_roc.result()

            print(f"Validation Current AUC ROC: {current_auc_roc}")

            if best_auc_roc < current_auc_roc:
                tf.print("Best AUC-ROC on validation set: ", current_auc_roc)

                # Replace the best
                best_auc_roc = current_auc_roc

                base_path = self._log_dir / self.dataset_name / "results" / "auc_roc"
                self.generator.summary()
                self.generator.save(
                    str(base_path / "generator"), overwrite=True)
                self.encoder.summary()
                self.encoder.save(str(base_path / "encoder"), overwrite=True)
                self.discriminator.summary()
                self.discriminator.save(
                    str(base_path / "discriminator"), overwrite=True
                )

                with open(base_path / "validation.json", "w", encoding='utf-8') as fp:
                    json.dump(
                        {
                            "value": float(best_auc_rc),
                        },
                        fp,
                    )

            self.auc_rc = self._auc_rc.result().numpy() 
            self.auc_roc = self._auc_roc.result().numpy()

            # Plot monitored performance metrics to TensorBoard
            self._report_performance_metrics(epoch=epoch)

            # Reset metrics or the data will keep accruing becoming an average of ALL the epochs
            self._reset_keras_metrics()

        return self.auc_rc, self.auc_roc, self._f1_value, self._f2_value, self._acc_value

    @tf.function
    def train_step(
        self,
        x,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
    ):

        # Random noise
        z = tf.random.normal((tf.shape(x)[0], self._hps["latent_vector_size"]), seed=constants.SEED)

        with tf.GradientTape(persistent=True) as tape:
            # Generator reconstruction from random noise
            g_z = self.generator(z, training=True)  

            # Discriminator on real data
            d_x = self.discriminator(x, training=True)

            # Reconstruct real data after encoding
            e_x = self.encoder(x, training=True)
            g_ex = self.generator(e_x, training=True)

            # Discriminator on the reconstructed real data g_ex
            d_gex = self.discriminator(inputs=g_ex, training=True)

            # Encode the reconstructed real data g_ex
            e_gex = self.encoder(g_ex, training=True)

            # Discriminator Loss
            # d_loss = self._minmax(d_x_features, d_gex_features)
            d_loss = self._minmax(d_x, d_gex)

            # Generator Loss
            # adversarial_loss = losses.adversarial_loss_fm(d_f_x, d_f_x_hat)
            bce_g_loss = generator_bce(g_ex, from_logits=True)

            l1_loss = self._mae(x, g_ex)  # Contextual loss
            e_loss = self._mse(e_x, e_gex)  # Encoder loss

            g_loss = (
                adversarial_loss_weight * bce_g_loss
                + contextual_loss_weight * l1_loss
                + enc_loss_weight * e_loss
            )

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        del tape

        self.optimizer_ge.apply_gradients(
            zip(
                g_grads,
                self.generator.trainable_variables + self.encoder.trainable_variables,
            )
        )
        self.optimizer_d.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )

        return (
            g_z,
            g_ex,
            d_loss,
            g_loss,
            e_loss,
        )

    def test(self, base_path: Union[Path, None] = None,
             test_dataset=None, experiment_name=None):
        """
        Test the model on all the meaningful metrics.

        Args:
            base_path: the path to use for loading the models. If None, the default is used.
            dataset: data used to the test the model
        """

        if test_dataset is None:
            print("[Info] Test dataset not provided, using models local data.")
            test_dataset = self._dataset.test

        if experiment_name is None:
            experiment_name = self._dataset.get_name()

        # Loop over every "best model" for every metric used in model selection
        for metric in ["auc_rc", "auc_roc"]:
            if not base_path:
                base_path = self._log_dir / self.dataset_name / "results" / metric
            encoder_path = base_path / "encoder"
            generator_path = base_path / "generator"

            # Load the best models to use as the model here
            encoder = k.models.load_model(encoder_path)
            generator = k.models.load_model(generator_path)

            # Resetting the state of the AUPRC variable
            self._auc_rc.reset_states()

            self._f1_value = 0.0
            self._f2_value = 0.0
            self._acc_value = 0.0

            tf.print("Using the best model selected via ", metric)
            # Test on the test dataset
            group_labels = list()
            group_scores = list()

            for batch in test_dataset:
                x, labels_test = batch
                # Get the anomaly scores
                anomaly_scores = self._compute_anomaly_scores(
                    x, encoder, generator)

                group_labels.extend(labels_test.numpy())
                group_scores.extend(anomaly_scores.numpy())

                self._auc_rc.update_state(labels_test, anomaly_scores)
                self._auc_roc.update_state(labels_test, anomaly_scores)

            fpr, tpr, _ = roc_curve(group_labels, group_scores)

            auc_rc = self._auc_rc.result()
            auc_roc = self._auc_roc.result()

            expected, anomaly = self._get_classified_data(group_labels, group_scores)

            self._f1_value = self._compute_f1_score(expected, anomaly)
            self._f2_value = self._compute_f2_score(expected, anomaly)
            self._acc_value = self._comput_accuracy(expected, anomaly) 

            tf.print(f"Experiment '{experiment_name}' Results")

            result_metrics = {
                "AUC-RC": auc_rc, 
                "AUC-ROC": auc_roc, 
                "F1-Score": self._f1_value, 
                "F2-Score": self._f2_value, 
                "Accuracy": self._acc_value
            }

            for ml_metric, value in result_metrics.items():
                tf.print(f"Test {ml_metric}: {value:.3f}")

            plot_roc_curve(experiment_name, fpr, tpr, result_metrics=result_metrics)

            generate_anomaly_plot(
                labels=group_labels, 
                anomaly_scores=group_scores, 
                threshold=self.anomaly_threshold, 
                title=self.dataset_name,
                save_to_file=True
            )

            base_path = self._log_dir / self.dataset_name / "results" / metric
            result = {
                "auc_roc": {
                    "value": float(auc_roc),
                },
                "auc_rc": {
                    "value": float(auc_rc),
                },
            }
            # Write the file
            with open(base_path / "test.json", "w", encoding="utf-8") as fp:
                json.dump(result, fp)

        return self._auc_rc.result().numpy(), self._auc_roc.result().numpy(), self._f1_value, self._f2_value, self._acc_value

    def _compute_anomaly_scores(
        self, x: tf.Tensor, encoder: k.Model, generator: k.Model
    ) -> tf.Tensor:
        """
        Compute the anomaly scores as indicated in the GANomaly paper
        https://arxiv.org/abs/1805.06725.

        Args:
            x: The batch of data to use to calculate the anomaly scores.

        Returns:
            The anomaly scores on the input batch, [0, 1] normalized.
        """

        # Get the generator reconstruction of a decoded input data
        e_x = encoder(x, training=False)

        g_ex = generator(e_x, training=False)

        # Encode the generated g_ex
        e_gex = encoder(g_ex, training=False)

        # Get the anomaly scores
        normalized_anomaly_scores, _ = tf.linalg.normalize(
            tf.norm(
                self._flatten(tf.abs(e_x - e_gex)),
                axis=1,
                keepdims=False,
            )
        )

        return normalized_anomaly_scores

    def _compute_youden_position(self, tpr, fpr):
        return np.argmax(tpr - fpr)
    
    def _compute_anomaly_score_threshold(self, anomaly_scores): 
        return np.percentile(anomaly_scores, 95)

    def _get_classified_data(self, expected_classes, anomaly_scores):
        expected = np.array(copy.copy(expected_classes)).astype(float)
        anomaly = np.array(copy.copy(anomaly_scores))

        anomaly[anomaly >= self.anomaly_threshold] = 1
        anomaly[anomaly < self.anomaly_threshold] = 0

        return expected, anomaly
    
    def _compute_f1_score(self, expected, anomaly):
        return f1_score(expected, anomaly)

    def _compute_f2_score(self, expected, anomaly):
        return fbeta_score(expected, anomaly, beta=2)
    
    def _comput_accuracy(self, expected, anomaly):
        return accuracy_score(expected, anomaly)
    
    def _report_performance_metrics(self, epoch): 
        with self._summary_writer.as_default():
            tf.summary.scalar(
                "auc_rc",
                self._auc_rc.result(),
                step=epoch
            )
            tf.summary.scalar(
                "auc_roc",
                self._auc_roc.result(),
                step=epoch
            )
