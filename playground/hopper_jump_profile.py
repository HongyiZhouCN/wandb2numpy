import matplotlib
import numpy as np
import os

import tikzplotlib
from matplotlib import pyplot as plt
import matplotlib

from wandb2numpy import util

matplotlib.use('TkAgg')
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def delete_jth_dim(array, j):
    return np.delete(array, j, axis=0)

def read_hopper_jump(data_path):
    subdict_list = get_immediate_subdirectories(data_path)

    height_list = []
    goal_distance_list = []
    reward_list = []
    global_steps_list = []
    iter = -1
    ignore_dim = 0
    for subdict in subdict_list:
        # Height
        height_path = subdict + "/evaluation_max_height.npy"
        height = np.load(height_path)
        height = height[:, :iter]
        # height = delete_jth_dim(height, ignore_dim)
        height_list.append(height)
        # Goal distance
        goal_distance_path = subdict + "/evaluation_goal_dist.npy"
        goal_distance = np.load(goal_distance_path)
        goal_distance = goal_distance[:, :iter]
        # goal_distance = delete_jth_dim(goal_distance, ignore_dim)
        goal_distance_list.append(goal_distance)
        # reward
        reward_path = subdict + "/evaluation_mean.npy"
        reward = np.load(reward_path)
        reward = reward[:, :iter]
        reward_list.append(reward)

        simulation_steps_path = subdict + "/simulation_steps.npy"
        simulation_steps = np.load(simulation_steps_path)
        simulation_steps = simulation_steps[:, :iter]
        # simulation_steps = delete_jth_dim(simulation_steps, ignore_dim)

        # col_vals = np.arange(iter) * 16000
        # simulation_steps = np.ones((20, 1)) * col_vals

        global_steps_list.append(simulation_steps)
    height_array = np.array(height_list)
    height_array = np.swapaxes(height_array, 0, 1)

    goal_distance_array = np.array(goal_distance_list)
    goal_distance_array = np.swapaxes(goal_distance_array, 0, 1)

    reward_array = np.array(reward_list)
    reward_array = np.swapaxes(reward_array, 0, 1)

    simulation_steps_array = np.array(global_steps_list)
    simulation_steps_array = np.swapaxes(simulation_steps_array, 0, 1)

    return height_array, goal_distance_array, reward_array, simulation_steps_array


def draw_hopper_iqm(profile_name, profile, simulation_steps, algorithm):
    # fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    num_frame = 20
    frames = np.floor(np.linspace(1, profile.shape[-1], num_frame)).astype(
        int) - 1

    profile = profile[:, None, :]
    profile = profile[:, :, frames]
    mask = np.isnan(profile)
    profile[mask] = 0.0
    frames_scores_dict = {algorithm: profile}
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., frame]) for frame in
         range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(frames_scores_dict, iqm,
                                                     reps=5000)
    plot_utils.plot_sample_efficiency_curve(simulation_steps[0, frames],
                                            iqm_scores, iqm_cis,
                                            algorithms=[algorithm],
                                            xlabel="Iteration", ylabel="IQM")
    tikzplotlib.get_tikz_code()
    tikzplotlib.save(f"hopper_jump_sac_{profile_name}_iqm.tex")
    plt.show()


if __name__ == "__main__":
    height, goal_distance, reward, simulation_steps = read_hopper_jump(
        f"/home/hongyi/Codes/bruce_iclr/wandb2numpy/wandb_data/hopper_sac_20seeds")

    reshaped_height = np.reshape(height, (-1, height.shape[-1]))
    reshaped_goal_distance = np.reshape(goal_distance, (-1, goal_distance.shape[-1]))
    reshaped_reward = np.reshape(reward, (-1, reward.shape[-1]))
    reshaped_simulation_steps = np.reshape(simulation_steps, (-1, simulation_steps.shape[-1]))

    profile_dict = {"height": reshaped_height,
                    "goal_distance": reshaped_goal_distance,
                    "reward": reshaped_reward}

    # Fixme, smoothness has been removed
    for key, profile in profile_dict.items():
        draw_hopper_iqm(key, profile, reshaped_simulation_steps, None)

























