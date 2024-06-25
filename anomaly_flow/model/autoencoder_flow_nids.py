from anomaly_flow.metrics.eval_learning import eval_learning

import os
import tensorflow as tf
import numpy as np 

from keras.layers import Input
from keras.layers import Dense

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class AutoEncoder():

    def __init__(self, num_features):
        self.num_features = num_features
        self.model = self.model()
        self.threshold = 0

    def model(self):

        model = tf.keras.models.Sequential([
            Input(shape=(self.num_features,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, activation="relu"),
            Dense(8, activation="relu"),
            Dense(16, activation="relu"),
            Dense(32, activation="relu"),
            Dense(self.num_features, activation="sigmoid")
        ]) 

        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def fit(self, train_data, epochs, batch_size, shuffle=True, dataset_name=None):

        x_train, y_train = train_data

        print("> Loaded", dataset_name if dataset_name is not None else "Unknown Dataset", " | Trainset:", x_train.shape)
        print("> Train samples:", y_train.shape[0])

        # Training only on benign traffic
        history = self.model.fit(
                x_train[y_train == 0], 
                x_train[y_train == 0], 
                epochs=epochs, 
                batch_size=batch_size, 
                shuffle=shuffle
        )

        self.threshold = history.history["loss"][-1]
        print(">>> Threshold:", self.threshold)
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, test_data, dataset_name=None):

        x_test, y_test = test_data

        print("> Loaded", dataset_name if dataset_name is not None else "Unknown Dataset", " | Testset:", x_test.shape)
        print("> Test samples:", y_test.value_counts().to_string().replace("\n", ", "))

        inference = self.model.predict(x_test)
        loss = self.model.evaluate(x_test, x_test)
        inference_loss = AutoEncoder.__calculate_reconstruction_loss(x_test, inference)

        y_pred = inference_loss > self.threshold

        acc, rec, prec, f1, mcc, missrate, fallout, auc, f2_value = eval_learning(y_test, y_pred)

        output_dict = {"acc": acc, "rec": rec, "prec": prec, "f1": f1, "mcc": mcc, "missrate": missrate,
                "fallout": fallout, "auc": auc, "f2-score": f2_value}

        print(f"Test Results:\n{output_dict}")

        return loss, len(x_test), output_dict

    def cross_evaluation(self, cross_test_data):
        x_validation, y_validation = cross_test_data

        inference = self.model.predict(x_validation) 
        loss = self.model.evaluate(x_validation, x_validation)
        inference_loss = AutoEncoder.__calculate_reconstruction_loss(x_validation, inference)
        y_pred = inference_loss > self.threshold
        acc, rec, prec, f1, mcc, missrate, fallout, auc, f2_value = eval_learning(y_validation, y_pred)

        output_dict = {"acc": acc, "rec": rec, "prec": prec, "f1": f1, "mcc": mcc, "missrate": missrate,
                "fallout": fallout, "auc": auc, "f2-score": f2_value}

        print(f"Cross-Evaluation Results:\n{output_dict}")

        return loss, len(x_validation), output_dict

    @staticmethod
    def __calculate_reconstruction_loss(x, x_hat):
        losses = np.mean(abs(x - x_hat), axis=1) 
        return losses