import numpy as np
import pandas
from rliable import plot_utils

data = np.load("ppo_sac_trpl.npy", allow_pickle=True)
time_steps = data[0]
iqm = data[1]
iqm_ci = data[2]

ppo_time = time_steps["ppo"]
sac_time = time_steps["sac"]
trpl_time = time_steps["trpl"]

keys = list(iqm.keys())
sac_iqm = {key: iqm[key] for key in keys[0:50]}
ppo_iqm = {key: iqm[key] for key in keys[50:100]}
trpl_iqm = {key: iqm[key] for key in keys[100:150]}

sac_iqm_ci_low = {key: iqm_ci[key][0] for key in keys[0:50]}
ppo_iqm_ci_low = {key: iqm_ci[key][0] for key in keys[50:100]}
trpl_iqm_ci_low = {key: iqm_ci[key][0] for key in keys[100:150]}

ppo_iqm_ci_high = {key: iqm_ci[key][1] for key in keys[0:50]}
sac_iqm_ci_high = {key: iqm_ci[key][1] for key in keys[50:100]}
trpl_iqm_ci_high = {key: iqm_ci[key][1] for key in keys[100:150]}

for i, (sc, cis) in enumerate(zip(element1.items(), element2.items())):
    algo, task = sc[0].split("_", maxsplit=1)
    plot_utils.plot_sample_efficiency_curve(
            np.abs(element3[algo]), dict([sc]), dict([cis]), algorithms=[sc[0]])

    print("")