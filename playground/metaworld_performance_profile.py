import numpy as np
import os

from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def read_meta_world_data(data_path):

    subdict_list = get_immediate_subdirectories(data_path)

    is_success_list = []
    global_steps_list = []

    for subdict in subdict_list:
        subdict = subdict + "/experiment1"
        if "evaluation_success_mean.npy" in os.listdir(subdict):
            is_success_path = subdict + "/evaluation_success_mean.npy"
            is_success = np.load(is_success_path)
            is_success_list.append(is_success)
        if "num_global_steps.npy" in os.listdir(subdict):
            simulation_steps_path = subdict + "/num_global_steps.npy"
            simulation_steps = np.load(simulation_steps_path)
            global_steps_list.append(simulation_steps)
    is_success_array = np.array(is_success_list)
    is_success_array = np.swapaxes(is_success_array, 0, 1)
    simulation_steps_array = np.array(global_steps_list)
    simulation_steps_array = np.swapaxes(simulation_steps_array, 0, 1)

    return is_success_array, simulation_steps_array

def draw_peformance_profile(is_success, x_points=30):
     is_success = is_success[:, :, -1]
     is_success = is_success[:, :, None]
     x_axis = np.linspace(0.0, 0.99, x_points)
     performance_dict = {"bbrl_prodmp": is_success}
     score_distributions, score_distribution_cis = rly.create_performance_profile(performance_dict, x_axis)
     fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
     plot_utils.plot_performance_profiles(score_distributions, x_axis, performance_profile_cis=score_distribution_cis,
                                         xlabel='Success Rate',
                                         ax=ax)
     plt.show()

def draw_metaworld_iqm(is_success, simulation_steps, algorithm):
    num_frame = 20
    frames = np.floor(np.linspace(1, is_success.shape[-1], num_frame)).astype(int) - 1

    is_success = is_success[:, None, :]
    is_success = is_success[:, :, frames]
    mask = np.isnan(is_success)
    is_success[mask] = 0.0
    frames_scores_dict = {algorithm: is_success}
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(frames_scores_dict, iqm, reps=5000)
    plot_utils.plot_sample_efficiency_curve(simulation_steps[0, frames], iqm_scores, iqm_cis,
                                            algorithms=[algorithm], xlabel="Iteration", ylabel="IQM")
    plt.show()

if __name__ == "__main__":
    is_success, simulation_steps = read_meta_world_data("/home/hongyi/Codes/alr_ma/wandb2numpy/wandb_data/metaworld_bbrl_prodmp")

    # draw the performance profile
    draw_peformance_profile(is_success)

    # draw the iqm curve
    reshaped_is_success = np.reshape(is_success, (-1, is_success.shape[-1]))
    reshaped_simulation_steps = np.reshape(simulation_steps, (-1, simulation_steps.shape[-1]))
    draw_metaworld_iqm(reshaped_is_success, reshaped_simulation_steps, "bbrl_prodmp")