"""
    Module used to train multiple External Machine Learning Models.
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb

from anomaly_flow.utils.binary_processing import split_flag_columns

APPLY_SYNTHETIC = True
APPLY_REAL = False
SAME_SCALE = True

dtypes_netflow = {
    "IPV4_SRC_ADDR":                "object",
    "L4_SRC_PORT":                  "float32",
    "IPV4_DST_ADDR":                "object",
    "L4_DST_PORT":                  "float32",
    "PROTOCOL":                     "float32",
    "L7_PROTO":                     "float64",
    "IN_BYTES":                     "float32",
    "IN_PKTS":                      "float32",
    "OUT_BYTES":                    "float32",
    "OUT_PKTS":                     "float32",
    "TCP_FLAGS":                    "int32",
    "CLIENT_TCP_FLAGS":             "int32",
    "SERVER_TCP_FLAGS":             "int32",
    "FLOW_DURATION_MILLISECONDS":   "float32",
    "DURATION_IN":                  "float32",
    "DURATION_OUT":                 "float32",
    "MIN_TTL":                      "float32",
    "MAX_TTL":                      "float32",
    "LONGEST_FLOW_PKT":             "float32",
    "SHORTEST_FLOW_PKT":            "float32",
    "MIN_IP_PKT_LEN":               "float32",
    "MAX_IP_PKT_LEN":               "float32",
    "SRC_TO_DST_SECOND_BYTES":      "float64",
    "DST_TO_SRC_SECOND_BYTES":      "float64",
    "RETRANSMITTED_IN_BYTES":       "float32",
    "RETRANSMITTED_IN_PKTS":        "float32",
    "RETRANSMITTED_OUT_BYTES":      "float32",
    "RETRANSMITTED_OUT_PKTS":       "float32",
    "SRC_TO_DST_AVG_THROUGHPUT":    "float32",
    "DST_TO_SRC_AVG_THROUGHPUT":    "float32",
    "NUM_PKTS_UP_TO_128_BYTES":     "float32",
    "NUM_PKTS_128_TO_256_BYTES":    "float32",
    "NUM_PKTS_256_TO_512_BYTES":    "float32",
    "NUM_PKTS_512_TO_1024_BYTES":   "float32",
    "NUM_PKTS_1024_TO_1514_BYTES":  "float32",
    "TCP_WIN_MAX_IN":               "float32",
    "TCP_WIN_MAX_OUT":              "float32",
    "ICMP_TYPE":                    "float32",
    "ICMP_IPV4_TYPE":               "float32",
    "DNS_QUERY_ID":                 "float32",
    "DNS_QUERY_TYPE":               "float32",
    "DNS_TTL_ANSWER":               "float32",
    "FTP_COMMAND_RET_CODE":         "float32",
    "Attack":                       "object",
    "Label":                        "float32",
}

FEATURES_TO_DROP = [
    'IPV4_SRC_ADDR', 
    'IPV4_DST_ADDR', 
    'L7_PROTO', 
    'L4_SRC_PORT', 
    'L4_DST_PORT', 
    'FTP_COMMAND_RET_CODE',
    'Attack'
]

# Load a sample dataset for demonstration (you can replace it with your own dataset)
df = pd.read_csv(
    "./datasets/NF-UNSW-NB15-v2-downsample.csv.gz",
    dtype=dtypes_netflow
)

cross_df = pd.read_csv(
    "./datasets/NF-ToN-IoT-v2-DDoS-downsample.csv.gz",
    dtype=dtypes_netflow
)

threshold = np.finfo(np.float32).max

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(df['Attack'].value_counts())
df.drop(FEATURES_TO_DROP, axis=1, inplace=True)
df = df[df < threshold]
df = split_flag_columns(df)
df.dropna(inplace=True)

cross_df.replace([np.inf, -np.inf], np.nan, inplace=True)
cross_df.dropna(inplace=True)
print(cross_df['Attack'].value_counts())
cross_df.drop(FEATURES_TO_DROP, axis=1, inplace=True)
cross_df = cross_df[cross_df < threshold]
cross_df = split_flag_columns(cross_df)
cross_df.dropna(inplace=True)

cross_x, cross_y = cross_df.drop(['Label', 'Unnamed: 0'], axis=1), cross_df['Label']

X, y = df.drop(['Label'], axis=1), df['Label']

X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, stratify=y,
                                        test_size=0.33, random_state=42
                                   )
cross_x_train, cross_x_test, cross_y_train, cross_y_test = train_test_split(
                                                                cross_x, cross_y,
                                                                stratify=cross_y,
                                                                test_size=0.9,
                                                                random_state=42
                                                           )

# Reescale the models to train and test
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if APPLY_SYNTHETIC is True:
    synthetic_df = pd.read_parquet(
        "./datasets/synthetic_ton_iot_ganomaly.parquet"
    )
    synthetic_df["Label"] = 0
    synthetic_x, synthetic_y = synthetic_df.drop(['Label'], axis=1), synthetic_df['Label']
    synthetic_x = synthetic_x.to_numpy()

    if SAME_SCALE is True:
        synthetic_x = scaler.transform(synthetic_x)
 
    X_train = np.concatenate((X_train, synthetic_x), axis=0)
    y_train = np.concatenate((y_train, synthetic_y))

if SAME_SCALE is True:
    cross_x_train = scaler.transform(cross_x_train)
    cross_x_test = scaler.transform(cross_x_test)
else:
    external_scaler = MinMaxScaler()
    cross_x_train = external_scaler.fit_transform(cross_x_train)

if APPLY_REAL is True:
    X_train = np.concatenate((X_train, cross_x_train), axis=0)
    y_train = np.concatenate((y_train, cross_y_train))



# Define parameter grids for each algorithm
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
}

param_grid_if = {
    'n_estimators': [50, 100, 150],
    'contamination': [0.01, 0.05, 0.1, 0.2]
}

param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Create instances of the classifiers
rf_classifier = RandomForestClassifier()
lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
if_classifier = IsolationForest(random_state=42)
xgb_classifier = xgb.XGBClassifier()

# Create a dictionary of classifiers and their respective parameter grids
classifiers = {
    'Random Forest': (rf_classifier, param_grid_rf, 'f1', False),
    'Logistic Regression': (lr_classifier, param_grid_lr, 'f1', False),
    'Isolation Forest': (if_classifier, param_grid_if, 'roc_auc', True), 
    'eXtreme Gradient Boosting': (xgb_classifier, param_grid_xgb, 'f1', False)
}

# Evaluate each classifier using GridSearchCV
for classifier_name, (classifier, param_grid, metric, probability) in classifiers.items():

    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        scoring=metric,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    # Print the best parameters and the corresponding accuracy on the test set
    print(f"Best parameters for {classifier_name}: {grid_search.best_params_}")

    if probability is False:
        # Make predictions on the test set
        y_pred = grid_search.predict(X_test)
        cross_pred = grid_search.predict(cross_x_test)

        # Evaluate the performance on the same silo
        accuracy_value = accuracy_score(y_test, y_pred)
        f1_value = f1_score(y_test, y_pred)
        f2_value = fbeta_score(y_test, y_pred, beta=2)

        print(f"Accuracy on the test set: {accuracy_value:.4f}")
        print(f"F1-Score on the test set: {f1_value:.4f}")
        print(f"F2-Score on the test set: {f2_value:.4f}")

        # Evaluate the performance on cross silos approach
        accuracy_cross = accuracy_score(cross_y_test, cross_pred)
        f1_cross = f1_score(cross_y_test, cross_pred)
        f2_cross = fbeta_score(cross_y_test, cross_pred, beta=2)

        print(f"Accuracy on the cross-evaluation set: {accuracy_cross:.4f}")
        print(f"F1-Score on the cross-evaluation set: {f1_cross:.4f}")
        print(f"F2-Score on the cross-evaluation set: {f2_cross:.4f}")

    elif probability is True:
        anomaly_scores = grid_search.decision_function(X_test)

        # Evaluate the perfomance on the local data samples
        roc_auc = roc_auc_score(y_test, -anomaly_scores)
        pr_auc = average_precision_score(y_test, -anomaly_scores)
        print(f"ROC AUC Score on the test set: {roc_auc:.4f}")
        print(f"PR AUC Score on the test set: {pr_auc:.4f}")

        # Evaluate the performance on cross silos approach
        anomaly_scores_cross = grid_search.decision_function(cross_x_test)
        roc_auc = roc_auc_score(cross_y_test, -anomaly_scores_cross)
        pr_auc = average_precision_score(cross_y_test, -anomaly_scores_cross)
        print(f"ROC AUC Score on the cross set: {roc_auc:.4f}")
        print(f"PR AUC Score on the cross set: {pr_auc:.4f}")
