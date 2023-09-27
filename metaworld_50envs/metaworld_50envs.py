from typing import Union

import numpy as np
from rliable import metrics
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib.ticker import MaxNLocator
env_id = [
    "Assembly",
    "PickOutOfHole",
    "PlateSlide",
    "PlateSlideBack",
    "PlateSlideSide",
    "PlateSlideBackSide",
    "BinPicking",
    "Hammer",
    "SweepInto",
    "BoxClose",
    "ButtonPress",
    "ButtonPressWall",
    "ButtonPressTopdown",
    "ButtonPressTopdownWall",
    "CoffeeButton",
    "CoffeePull",
    "CoffeePush",
    "DialTurn",
    "Disassemble",
    "DoorClose",
    "DoorLock",
    "DoorOpen",
    "DoorUnlock",
    "HandInsert",
    "DrawerClose",
    "DrawerOpen",
    "FaucetOpen",
    "FaucetClose",
    "HandlePressSide",
    "HandlePress",
    "HandlePullSide",
    "HandlePull",
    "LeverPull",
    "PegInsertSide",
    "PickPlaceWall",
    "Reach",
    "PushBack",
    "Push",
    "PickPlace",
    "PegUnplugSide",
    "Soccer",
    "StickPush",
    "StickPull",
    "PushWall",
    "ReachWall",
    "ShelfPlace",
    "Sweep",
    "WindowOpen",
    "WindowClose",
    "Basketball",
]

COLOR_TCP = "1f77b4"  # % Blue, TCP
COLOR_BBRL = "17becf"  # % Cyan, BBRL
COLOR_PPO = "2ca02c"  # % Green, PPO
COLOR_TRPL = "d62728"  # % Red, TRPL
COLOR_SAC = "9467bd"  # % Purple, SAC
COLOR_gSDE = "8c564b"  # % Brown, gSDE
COLOR_PINK = "e377c2"  # % Pink, PINK


# \definecolor{C7}{HTML}{7f7f7f}  % Gray,
# \definecolor{C8}{HTML}{bcbd22}  % Yellow Green,
# \definecolor{C9}{HTML}{17becf}  % Cyan,
# \definecolor{C10}{HTML}{7fbc41} % Lime,

########################################################################
# Plot

def fill_between(x,
                 y_mean,
                 y_low,
                 y_high,
                 axis=None,
                 alpha=0.2, color='gray',
                 linewidth=2):
    """
    Utilities to draw std plot
    Returns:
        None
    """
    # x, y_mean, y_std = util.to_nps(x, y_mean, y_std)
    if axis is None:
        axis = plt.gca()

    axis.plot(x, y_mean, color=color, linewidth=linewidth)
    axis.fill_between(x=x,
                      y1=y_low,
                      y2=y_high,
                      alpha=alpha, color=color)


########################################################################
# gSDE
gsde_meta = np.load("meta_gsde.npy", allow_pickle=True).item()


def convert_task_id_gsde(input_str):
    output_str = []
    for i, char in enumerate(input_str):
        if char.isupper():
            if i != 0:
                output_str.append('-')
            output_str.append(char.lower())
        else:
            output_str.append(char)
    return ''.join(output_str) + "-v2"


def get_gsde_data(task_id):
    task_id_gsde = convert_task_id_gsde(task_id)
    # print(task_id_gsde)
    gsde_data = gsde_meta[task_id_gsde]
    return gsde_data


########################################################################
def convert_task_id_pink(input_str):
    return convert_task_id_gsde(input_str)


def get_pink_data(task_id):
    task_id_gsde = convert_task_id_pink(task_id)
    # print(task_id_gsde)
    # pink_data = np.load(
    #     f"./meta_pink/{task_id_gsde}/evaluation_success_mean.npy",
    #     allow_pickle=True)
    pink_data = np.load(
        f"./meta_pink/{task_id_gsde}/iqm.npy",
        allow_pickle=True)
    # Turn Nan to 0
    # pink_data[np.isnan(pink_data)] = 0
    return pink_data[:, 0], pink_data[:, 1], pink_data[:, 2], pink_data[:, 3]


########################################################################
# TCP
def convert_task_id_tcp(input_str):
    return input_str + "ProDMPTCP-v2"


def get_tcp_data(task_id):
    task_id_tcp = convert_task_id_tcp(task_id)

    success = np.load(
        f"./metaworld_tcp_prodmp/{task_id_tcp}/experiment1/evaluation_success_mean.npy",
        allow_pickle=True)
    time_steps = np.load(
        f"./metaworld_tcp_prodmp/{task_id_tcp}/experiment1/num_global_steps.npy",
        allow_pickle=True)
    # Turn Nan to 0
    success[np.isnan(success)] = 0
    time_steps[np.isnan(time_steps)] = 0
    return time_steps, success


########################################################################
# BBRL
def convert_task_id_bbrl(input_str):
    return input_str + "ProDMP-v2"


def get_bbrl_data(task_id):
    task_id_bbrl = convert_task_id_bbrl(task_id)

    success = np.load(
        f"./metaworld_bbrl_prodmp/{task_id_bbrl}/experiment1/evaluation_success_mean.npy",
        allow_pickle=True)
    time_steps = np.load(
        f"./metaworld_bbrl_prodmp/{task_id_bbrl}/experiment1/num_global_steps.npy",
        allow_pickle=True)
    # Turn Nan to 0
    success[np.isnan(success)] = 0
    time_steps[np.isnan(time_steps)] = 0
    return time_steps, success


########################################################################
def plot_main():
    # task_ids = env_id[10:12]
    task_ids = env_id
    for i, task_id in enumerate(task_ids):
        print(task_id)
        fig = plt.figure(figsize=(6, 4))
        ###################  PINK
        pink_time, pink_iqm, pink_low, pink_high = get_pink_data(task_id)  #
        fill_between(pink_time.T, pink_iqm.T, pink_low.T, pink_high.T, plt.gca(),
                 0.2, color=f'#{COLOR_PINK}')

        ##################  gSDE
        gsde_data = get_gsde_data(task_id) # [20, 100], 0-40M
        gsde_time = np.linspace(0, 40e6, gsde_data.shape[1])
        gsde_time = np.tile(gsde_time, (gsde_data.shape[0], 1))
        gsde_times, gsde_iqm, gsde_ci_low, gsde_ci_up = compute_iqm(gsde_time,
                                                                gsde_data)
        fill_between(gsde_times, gsde_iqm, gsde_ci_low, gsde_ci_up, plt.gca(),
                 0.2, color=f'#{COLOR_gSDE}')  # todo color

        ##################  BBRL
        bbrl_time, bbrl_data = get_bbrl_data(task_id)  # [20, 300], [20, 300]
        bbrl_times, bbrl_iqm, bbrl_ci_low, bbrl_ci_up = compute_iqm(bbrl_time,
                                                                    bbrl_data)
        fill_between(bbrl_times, bbrl_iqm, bbrl_ci_low, bbrl_ci_up, plt.gca(),
                     0.2, color=f'#{COLOR_BBRL}')  # todo color

        ####################  TCP
        tcp_times, tcp_data = get_tcp_data(task_id)  # [20, 100], 0-40M
        tcp_times, tcp_iqm, tcp_ci_low, tcp_ci_up = compute_iqm(tcp_times,
                                                                    tcp_data)
        fill_between(tcp_times, tcp_iqm, tcp_ci_low, tcp_ci_up, plt.gca(),
                     0.2, color=f'#{COLOR_TCP}', linewidth=5)  # todo color
        plt.xlim(0, 4e7)
        plt.ylim(0, 1)

        plt.xlabel("Number of Env Interaction", fontsize=20)
        plt.ylabel("Success rate IQM", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(f"{task_id}", fontsize=20)

        # Reduce the number of x-axis ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        plt.grid(alpha=0.5)

        # plt.show()

        print("")
        plt.savefig(f"./plots_50/plot_{i}.pdf", dpi=100, bbox_inches="tight")
        plt.close(fig)


def compute_iqm(simulation_steps, is_success):
    num_frame = 20
    frames = np.floor(np.linspace(1, is_success.shape[-1], num_frame)).astype(
        int) - 1

    is_success = is_success[:, None, :]
    is_success = is_success[:, :, frames]
    mask = np.isnan(is_success)
    is_success[mask] = 0.0
    frames_scores_dict = {"algorithm": is_success}
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., frame]) for frame in
         range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(frames_scores_dict, iqm,
                                                     reps=5000)
    return simulation_steps[0, frames], iqm_scores["algorithm"], \
           iqm_cis["algorithm"][0], iqm_cis["algorithm"][1]
    # plot_utils.plot_sample_efficiency_curve(simulation_steps[0, frames], iqm_scores, iqm_cis,
    #                                         algorithms=["algorithm"], xlabel="Iteration", ylabel="IQM")


if __name__ == "__main__":
    plot_main()
