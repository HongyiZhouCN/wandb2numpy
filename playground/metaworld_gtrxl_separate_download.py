from wandb2numpy.config_loader import load_config
from wandb2numpy.export import export_data
from wandb2numpy.save_experiment import create_output_dirs, save_matrix

if __name__ == "__main__":
    default_config = "/home/hongyi/Codes/bruce_iclr25/wandb2numpy/example_configs/metaworld_gtrxl.yaml"
    env_name_list = ["assembly-v2",
         "pick-out-of-hole-v2",
         "plate-slide-v2",
         "plate-slide-back-v2",
         "plate-slide-side-v2",
         "plate-slide-back-side-v2",
         "bin-picking-v2",
         "hammer-v2",
         "sweep-into-v2",
         "box-close-v2",
         "button-press-v2",
         "button-press-wall-v2",
         "button-press-topdown-v2",
         "button-press-topdown-wall-v2",
         "coffee-button-v2",
         "coffee-pull-v2",
         "coffee-push-v2",
         "dial-turn-v2",
         "disassemble-v2",
         "door-close-v2",
         "door-lock-v2",
         "door-open-v2",
         "door-unlock-v2",
         "hand-insert-v2",
         "drawer-close-v2",
         "drawer-open-v2",
         "faucet-open-v2",
         "faucet-close-v2",
         "handle-press-side-v2",
         "handle-press-v2",
         "handle-pull-side-v2",
         "handle-pull-v2",
         "lever-pull-v2",
         "peg-insert-side-v2",
         "pick-place-wall-v2",
         "reach-v2",
         "push-back-v2",
         "push-v2",
         "pick-place-v2",
         "peg-unplug-side-v2",
         "soccer-v2",
         "stick-push-v2",
         "stick-pull-v2",
         "push-wall-v2",
         "reach-wall-v2",
         "shelf-place-v2",
         "sweep-v2",
         "window-open-v2",
         "window-close-v2",
         "basketball-v2" ]

    list_doc = load_config(default_config)
    root_output_path = list_doc['experiment1']['output_path']
    for env_name in env_name_list:
        list_doc['experiment1']['config']['environment.env_id']['values'][
            0] = env_name
        list_doc['experiment1'][
            'output_path'] = root_output_path + "/" + env_name
        experiment_data_dict, config_list = export_data(list_doc)
        for i, experiment in enumerate(experiment_data_dict.keys()):
            experiment_dir = create_output_dirs(config_list[i], experiment)
            print(experiment_dir)

            for field in experiment_data_dict[experiment]:
                save_matrix(experiment_data_dict[experiment], experiment_dir,
                            field, True, config_list[i])
