"""
    Module used to plot data distributions.
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from anomaly_flow.metrics.wasserstein import calculate_wasserstein

def generate_dist_plot(series, comparison_series, title, save_to_file=False, ax=None):
    """
        Function to plot distribution of different series.
    """

    plt.figure(figsize=(10, 6))

    df_series = pd.DataFrame()
    df_series['data_column'] = series
    df_series['traffic_origin'] = 'real'

    df_comparison_series = pd.DataFrame()
    df_comparison_series['data_column'] = comparison_series
    df_comparison_series['traffic_origin'] = 'synthetic'

    df_plot = pd.concat([df_series, df_comparison_series], axis=0)

    del df_series
    del df_comparison_series

    current_plot_dir = "./plots/dist_plots"
    wasserstein_calculated = calculate_wasserstein(series, comparison_series)

    ax = sns.displot(df_plot, x='data_column', hue="traffic_origin", kind="kde", ax=ax)
    ax.set(title=f"Distribution Visualization \n {title}")
    ax.fig.text(
        x=0.5,
        y=1.05,
        s=f"Wasserstein Coefficient between distributions: {wasserstein_calculated}",
        fontsize=8,
        alpha=0.75,
        ha='center',
        va='bottom', bbox=dict(facecolor='white', alpha=0.5))

    if save_to_file is True:
        if not os.path.isdir(current_plot_dir):
            os.makedirs(current_plot_dir)
        plt.tight_layout()
        plt.savefig(f"{current_plot_dir}/{title}", bbox_inches="tight")
    else:
        plt.show()


def generate_anomaly_plot(labels, anomaly_scores, threshold, title, save_to_file=False):

    plot_df = pd.DataFrame({'labels': labels, 'anomaly_scores': anomaly_scores})
    
    current_plot_dir = "./plots/anomaly_plots"

    
    plt.figure(figsize=(8, 6))
    ax = sns.displot(plot_df, x='anomaly_scores', hue="labels")
    ax.set(title=f"Anomaly Scores \n {title}", xlim=(0, max(anomaly_scores)), yscale='log')
    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1)

    if save_to_file is True:
        if not os.path.isdir(current_plot_dir):
            os.makedirs(current_plot_dir)
        plt.tight_layout()
        plt.savefig(f"{current_plot_dir}/{title}", bbox_inches="tight")
        plt.show()
    else:
        plt.show()