import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
# Create a custom legend
from matplotlib.patches import Patch


def process_data(datadir, datadir2,args):
    # Step 1: Load the data
    csv_files = os.path.join(datadir, "experiment_log.csv")
    csv_files2 = os.path.join(datadir2, "experiment_log.csv")
    data = pd.read_csv(csv_files)
    data2 = pd.read_csv(csv_files2)

    # Sample loading of the DataFrame
    # data = pd.read_csv('your_data.csv')

    # Identify the unique severity levels and the metrics of interest
    unique_severities = data['severity'].unique()

    # Setup for plotting
    metrics = ['ag0_mean', 'ag0_std', 'ag0_rmse', 'ag1_mean', 'ag1_std', 'ag1_rmse']
    # for severity in unique_severities:
    #     vals = []
    #     severity_data = data[data['severity'] == severity]
    #     for metric in metrics:
    #         average_data = severity_data[metrics].mean()
    #         vals.append(average_data)
    #     print(f"Average data for severity {severity}:\n{vals}")
    n_metrics = len(metrics)
    n_severities = len(unique_severities)
    bar_width = 0.03
    opacity = 0.8
    gap = 0.05

    # Create figure and axes
    # fig, ax = plt.subplots(figsize=(15, 8))

    # # Calculate the total width for all bars in a group
    # total_width = n_severities * bar_width

    # for i, severity in enumerate(unique_severities):
    #     vals = []
    #     vals2 = []
    #     severity_data = data[data['severity'] == severity]
    #     severity_data2 = data2[data2['severity'] == severity]
    #     for j, metric in enumerate(metrics):
    #         average_data = severity_data[metric].mean()
    #         average_data2 = severity_data2[metric].mean()
    #         vals.append(average_data)
    #         vals2.append(average_data2)
    #     print(f"Average data for severity {severity}:\n{vals}")
    #     # Set the x position of the group of bars
    #     pos1 = j * (n_severities * bar_width + gap) + i * bar_width
    #     pos2 = pos1 + bar_width

    #     # Create a bar with pre_score data,
    #     # in position pos,
    #     plt.bar(pos1, vals, bar_width,
    #             alpha=opacity,
    #             color=plt.cm.viridis(0.3),
    #             label=f'Data Set 1 - Severity {severity}' if i == 0 and j == 0 else '')

    #     plt.bar(pos2, vals2, bar_width,
    #             alpha=opacity,
    #             color=plt.cm.viridis(0.7),
    #             label=f'Data Set 2 - Severity {severity}' if i == 0 and j == 0 else '')

    # metric_positions = [i * (n_severities * bar_width + gap) + (n_severities * bar_width / 2) for i in range(n_metrics)]
    # # Customize the x-axis
    # plt.xticks(range(n_metrics), metrics)

    # # Adding labels and title
    # plt.xlabel('Metrics')
    # plt.ylabel('Mean Values')
    # plt.title('Effect of Severity on Metrics')

    # legend_elements = [Patch(label=f'Severity {s}', color=plt.cm.viridis((s-1)/(n_severities-1))) for s in unique_severities]
    # plt.legend(handles=legend_elements, title='Severities')

    # plt.tight_layout()
    # plt.show()
    fig, ax = plt.subplots(figsize=(15, 8))

    for i, metric in enumerate(metrics):
        for j, severity in enumerate(unique_severities):
            vals1 = data[data['severity'] == severity][metric].mean()
            vals2 = data2[data2['severity'] == severity][metric].mean()

            # Calculate positions for the two bars
            pos1 = i * (n_severities * bar_width + gap) + j * bar_width
            pos2 = pos1 + bar_width

            # Plot bars
            plt.bar(pos1, vals1, bar_width, alpha=opacity, color=plt.cm.viridis(0.3), label=f'{args.exp} - Severity {severity}' if i==0 and j==0 else "")
            plt.bar(pos2, vals2, bar_width, alpha=opacity, color=plt.cm.viridis(0.6), label=f'{args.exp2} - Severity {severity}' if i==0 and j==0 else "")

    # Customizing the x-axis
    metric_positions = [i * (n_severities * bar_width + gap) + (n_severities * bar_width / 2) for i in range(n_metrics)]
    plt.xticks(metric_positions, metrics)

    # Adding labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Mean Values')
    plt.title('Comparison of Metric Means Across Severities for Two Data Sets')

    # Create a legend to distinguish between datasets
    legend_elements = [
        Patch(facecolor=plt.cm.viridis(0.3), label=args.exp),
        Patch(facecolor=plt.cm.viridis(0.6), label=args.exp2)
    ]
    plt.legend(handles=legend_elements, title='Experiments')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    datadir = "/home/lidonghao/ws/covins_ws/src/covins/covins_backend/output"
    parser = argparse.ArgumentParser(description="Process data from the perturbation experiment")
    parser.add_argument("--exp", type=str, help="Experiment name", default="perturb_agent0_severity1_10_trails_exp")
    parser.add_argument("--exp2", type=str, help="Experiment name", default="perturb_agent0_severity2_10_trails_exp")
    args = parser.parse_args()

    result_dir = os.path.join(datadir, args.exp)
    result_dir2 = os.path.join(datadir, args.exp2)
    process_data(result_dir, result_dir2,args)
