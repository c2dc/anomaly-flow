"""
    Module specified to plot ROC curves and Calculate different parameter types.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve(experiment_name, fpr, tpr, result_metrics, plot_to_file=True):
    """
        Method to plot ROC curves using the MatplotLib library.
    """
    current_plot_dir = f"./plots/{experiment_name}"

    label=str()

    for metric, value in result_metrics.items():
        label = label + f"{metric}: {value:.3f}\n"

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {experiment_name}')
    plt.legend(loc='lower right')

    if(plot_to_file):
        if not os.path.isdir(current_plot_dir):
            os.makedirs(current_plot_dir)
        plt.savefig(f"{current_plot_dir}/roc_curve")
        plt.show()
    else:
        plt.show()

    np.save(f"{current_plot_dir}/{experiment_name}_fpr.npy", fpr)
    np.save(f"{current_plot_dir}/{experiment_name}_tpr.npy", tpr)
