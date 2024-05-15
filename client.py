"""
    Federated learning client used to create an local model. 
"""
import os
import argparse
import json
import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from pathlib import Path
from anomaly_flow.data.netflow import NetFlowV2
from anomaly_flow.train.trainer_flow_nids import GANomaly


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

hps = dict()

with open('hps.json', 'r', encoding='utf-8') as file:
    hps = json.load(file)


def calculate_reconstruction_loss(x, x_hat):
    """
        Function used to calculate the reconstruction loss of an AutoEncoder.
    """
    losses = np.mean(abs(x - x_hat), axis=1)  # MAE
    return losses


def eval_learning(y_test, preds):
    """
        Function used to calculate different leaning metrics.
    """
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    missrate = fn / (fn + tp)
    fallout = fp / (fp + tn)
    auc = roc_auc_score(y_test, preds)
    return acc, rec, prec, f1, mcc, missrate, fallout, auc


class FlwrClient(fl.client.NumPyClient):
    """
        Class used to train the anomaly_flow model using the Flower framework. 
    """
    def __init__(self, dataset, specified_train_size=None):
        self._dataset = dataset
        self.specified_train_size = specified_train_size
        self.model = self.model()

    def create_model_trainer(self):
        """
            Method used to create the local model for this client.
        """
        self.netflow_dataset = NetFlowV2(self._dataset.split('.')[0], self.specified_train_size)

        self.netflow_dataset.configure(
            hps["batch_size"], 52, 1,
            hps["shuffle_buffer_size"], True, True
        )

        return GANomaly(
            self.netflow_dataset,
            hps,
            tf.summary.create_file_writer("logs"),
            Path("log")
        )

    def model(self):
        """
            Method to define the model trainer.
        """
        self.netflow_trainer = self.create_model_trainer()
        return self.netflow_trainer.get_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
            Method used to train the model.
        """
        self.model.set_weights(parameters)

        # Training the anomaly model only on benign traffic
        auc_rc, auc_roc, f1_value, f2_value, acc_value = self.netflow_trainer.train(
            hps["epochs"], hps["adversarial_loss_weight"], hps["contextual_loss_weight"],
            hps["enc_loss_weight"], hps["step_log_frequency"]
        )

        output_dict = {
            "auc_rc": auc_rc.item(), 
            "auc_roc": auc_roc.item(), 
            "f1_value": f1_value, 
            "f2_value": f2_value, 
            "acc_value": acc_value,
        }

        return self.model.get_weights(), self.netflow_dataset.get_train_size(), output_dict

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        auc_rc, auc_roc, f1_value, f2_value, acc_value = self.netflow_trainer.test()

        output_dict = {
            "auc_rc": auc_rc.item(), 
            "auc_roc": auc_roc.item(), 
            "f1_value": f1_value, 
            "f2_value": f2_value, 
            "acc_value": acc_value
        }

        print(output_dict)

        with open(f"./log/{self._dataset}/federated_results.json", "w") as outfile: 
            json.dump(output_dict, outfile)
            
        return 0.0, self.netflow_dataset.get_test_size().item(), output_dict


def main():
    """
        Main class used to define the Flower client during training.
    """
    parser = argparse.ArgumentParser(description="Netflow Flower Client")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_size", type=int, required=False)
    args = parser.parse_args()
    print(args.dataset)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081", client=FlwrClient(args.dataset, args.train_size))


if __name__ == "__main__":
    main()
