"""
    Federated Learning client using the Flower framework.
"""
from pathlib import Path
import tensorflow as tf
import flwr as fl
import json

from anomaly_flow.data.netflow import NetFlowV2
from anomaly_flow.train.trainer_flow_nids import GANomaly

hps = dict()

with open('hps.json', 'r', encoding='utf-8') as file:
    hps = json.load(file)

NUM_ROUNDS = 10

def average_metrics(metrics):
    """
        Function to calculate the average metrics for the clients. 
    """
    auc_rcs = [metric["auc_rc"] for _, metric in metrics]
    auc_rocs = [metric["auc_roc"] for _, metric in metrics]
    f1_values = [metric["f1_value"] for _, metric in metrics]
    f2_values = [metric["f2_value"] for _, metric in metrics]
    acc_values = [metric["acc_value"] for _, metric in metrics]

    auc_rcs = sum(auc_rcs) / len(auc_rcs)
    auc_rocs = sum(auc_rocs) / len(auc_rocs)
    f1_values = sum(f1_values) / len(f1_values)
    f2_values = sum(f2_values) / len(f2_values)
    acc_values = sum(acc_values) / len(acc_values)

    return {
        "auc_rc": auc_rcs, 
        "auc_roc": auc_rocs, 
        "f1_score": f1_values, 
        "f2_score": f2_values,
        "accuracy": acc_values
    }


def create_model_trainer():
    """
        Function to create a dummy centralized model for Federated Learning Schema. 
    """
    netflow_dataset = NetFlowV2("NF-CSE-CIC-IDS2018-v2-DDoS-downsample", train_size=5000)

    netflow_dataset.configure(
        hps["batch_size"], 52, 1,
        hps["shuffle_buffer_size"], True, True
    )

    return GANomaly(
        netflow_dataset,
        hps,
        tf.summary.create_file_writer("logs"),
        Path("log")
    )


def pretrain():
    """
        Pre train the model to initialize default weights. 
    """
    netflow_trainer = create_model_trainer()
    model = netflow_trainer.get_model()
    weights = model.get_weights()
    return weights


def main():
    """
        Creates the default main function for the Federated Learning Schema. 
    """
    print(">>> Flower version:", fl.__version__)

    strategy_1 = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=average_metrics,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=fl.common.ndarrays_to_parameters(pretrain())
    )
    
    # strategy_2 = fl.server.strategy.QFedAvg(
    #     evaluate_metrics_aggregation_fn=average_metrics,
    #     min_fit_clients=3,
    #     min_available_clients=3,
    #     initial_parameters=fl.common.ndarrays_to_parameters(pretrain())
    # )

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        strategy=strategy_1,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    )


if __name__ == "__main__":
    main()
