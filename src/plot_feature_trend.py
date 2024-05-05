import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
# Create a custom legend
from matplotlib.patches import Patch


def process_data(datadir):
    txt = os.path.join(datadir, "LOGperImageInput.txt")
    # timestamp, filename, tracking state, num key points, num map points
    if not os.path.exists(txt):
        print(f"File {txt} does not exist.")
        return
    df = pd.read_csv(txt, header=None, sep=",", names=['timestamp', 'id', 'state', 'number_of_key_points', 'number_of_map_points'])

    df = df[df['id'] > 100]
    mean_keypoints = np.mean(df['number_of_key_points'])
    std_keypoints = np.std(df['number_of_key_points'])
    print(f"Mean number of key points in state 1: {mean_keypoints}, std: {std_keypoints}")
    plt.figure(figsize=(10, 6))
    plt.plot(df['id'], df['number_of_key_points'], marker='o', linestyle='-', color='b', linewidth=0.5, markersize=4)

    state_one_ids = df[df['state'] == 1]['id']
    state_one_keypoints = df[df['state'] == 1]['number_of_key_points']
    plt.scatter(state_one_ids, state_one_keypoints, color='r')
    plt.title('Trend of Number of Key Points by ID')
    plt.xlabel('ID')
    plt.ylabel('Number of Key Points')
    plt.grid(True)
    plt.savefig(os.path.join(datadir, "keypoints_trend.png"))
    # plt.show()

if __name__ == "__main__":
    datadir = "/home/lidonghao/ws/covins_ws/src/ablation_perturbation/results/EuRoC/agn0/perturbed_results"
    parser = argparse.ArgumentParser(description="Process data from the perturbation experiment")
    parser.add_argument("--exp", type=str, default="denoised_perturb_bag_s1_f500_p100")
    parser.add_argument("--trail", type=str, default="n0_lv1_exp1")
    args = parser.parse_args()
    # trail_name = "{}_n0_lv{}_exp1".format(args.n, args.severity)
    print(f"Processing data for {args.exp} trail {args.trail}")
    result_dir = os.path.join(datadir, args.exp, args.trail)
    process_data(result_dir)
