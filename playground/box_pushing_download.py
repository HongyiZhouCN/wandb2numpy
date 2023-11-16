from wandb2numpy.config_loader import load_config
from wandb2numpy.export import export_data
from wandb2numpy.save_experiment import create_output_dirs, save_matrix


if __name__=="__main__":
    # BBRL Dense
    # default_config = "/home/lige/Codes/wandb2numpy/example_configs/box_dense_bbrl.yaml"
    # TCP Dense
    # default_config = "/home/lige/Codes/wandb2numpy/example_configs/box_dense_tcp.yaml"

    # BBRL T Sparse
    # default_config = "/home/lige/Codes/wandb2numpy/example_configs/box_temporal_sparse_bbrl.yaml"
    # TCP T Sparse
    default_config = "/home/hongyi/Codes/bruce_iclr/wandb2numpy/example_configs/box_pushing_ppo.yaml"

    list_doc = load_config(default_config)
    experiment_data_dict, config_list = export_data(list_doc)
    print(experiment_data_dict.keys())
    for i, experiment in enumerate(experiment_data_dict.keys()):
        experiment_dir = create_output_dirs(config_list[i], experiment)
        print(experiment_dir)

        for field in experiment_data_dict[experiment]:
            save_matrix(experiment_data_dict[experiment], experiment_dir, field, True, config_list[i])