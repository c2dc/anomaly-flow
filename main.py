"""
    Class used to run local experiments using GANs and different Netflow datasets.
"""
import os
from pathlib import Path
import json
import argparse
import tensorflow as tf
from anomaly_flow.data.netflow import NetFlowV2
from anomaly_flow.train.trainer_flow_nids import GANomaly

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

parser = argparse.ArgumentParser(description='Anomaly Flow - Netflow Experiments.')
parser.add_argument('--train-dataset', type=str, help="Dataset to train the models.", required=True)
parser.add_argument("--train_size", type=int, required=False)
parser.add_argument(
    '--test-datasets', type=str,
    nargs='+', help="List of datasets to test the trained model."
)
args = parser.parse_args()


hps = dict()

with open('hps.json', 'r', encoding='utf-8') as file:
    hps = json.load(file)

if __name__ == "__main__":

    netflow_dataset = NetFlowV2(args.train_dataset)

    netflow_dataset.configure(
        hps["batch_size"], 52, 1,
        hps["shuffle_buffer_size"], True, True
    )

    netflow_trainer = GANomaly(
        netflow_dataset, hps, tf.summary.create_file_writer("logs"), Path("log")
    )

    netflow_trainer.train(
        hps["epochs"], hps["adversarial_loss_weight"], hps["contextual_loss_weight"],
        hps["enc_loss_weight"], hps["step_log_frequency"]
    )

    netflow_trainer.test()

    for dataset in args.test_datasets:
        cross_eval_dataset = NetFlowV2(dataset)

        cross_eval_dataset.configure(
            hps["batch_size"], 52, 1,
            hps["shuffle_buffer_size"], True, True
        )

        netflow_trainer.test(test_dataset=cross_eval_dataset.get_test_dataset(),
                            experiment_name=f"Cross_Evaluation_{args.train_dataset}-{dataset}")

        del cross_eval_dataset
