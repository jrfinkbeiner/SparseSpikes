import sys
import itertools
import numpy as np
import pandas as pd 
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


ALL_ARGS = [
    "use_sparse",
    "batchsize",
    "lr",
    "use_lr_scheduler",
    "second_threshold",
    "dataset_name",
    "max_activity",
    "num_hidden_layers",
    "sparse_size_inp",
    "seq_len",
    "num_epochs",
    "batchsize_test",
    "use_wandb",
    "use_bias",
    "use_lsuv",
    "act_rate",
    "ro_type",
    "ro_int",
    "use_thresh_scheduler",
    "thresh_scheduler_mul",
    "use_aug",
    "use_crop",
]

RELEVANT_ARGS_DENSE = [
    # "use_sparse",
    "batchsize",
    # "lr",
    "use_lr_scheduler",
    # "second_threshold",
    # "dataset_name",
    # "max_activity",
    "num_hidden_layers",
    # "sparse_size_inp",
    # "seq_len",
    # "num_epochs",
    # "batchsize_test",
    # "use_wandb",
    "use_bias",
    "use_lsuv",
    # "act_rate",
    "ro_type",
    # "ro_int",
    # "use_thresh_scheduler",
    # "thresh_scheduler_mul",
    "use_aug",
    "use_crop",
]

RELEVANT_ARGS_SPARSE = [
    *RELEVANT_ARGS_DENSE,
    "use_sparse",
    "max_activity",
    "sparse_size_inp",
    "second_threshold",
    "use_thresh_scheduler",
    "thresh_scheduler_mul",
]



def plot_data_and_convolved(ax, data, kernel_size, *, label, alpha=0.2, **kwargs):
    ax.plot(data, alpha=alpha, **kwargs)
    ax.plot(np.arange((kernel_size)//2, len(data)-kernel_size//2), np.convolve(data, np.ones((kernel_size,))/kernel_size, mode='valid'), label=label, **kwargs)


def plot_acc_comparison(ax, dense_history, sparse_histories, kernel_size=5, create_labels=True):
    for metric, axi in zip(["train/acc", "test/acc"], ax):
        axi.hlines(1.0, 0, len(dense_history[metric]), color="black", linestyle="dashed", label="dense" if create_labels else None)
        for i, (k, v) in enumerate(sparse_histories.items()):
            plot_data = v[metric]/dense_history[metric]
            plot_data_and_convolved(axi, plot_data, kernel_size, label=f"sparse {k}" if create_labels else None, alpha=0.2, color=f"C{i}")
        axi.set_title(metric)
        axi.legend()
        axi.set_xlabel("epoch")


def plt_abs_accuracies(ax, dense_history, sparse_histories, kernel_size=5, create_labels=True):
    for metric, axi in zip(["train/acc", "test/acc"], ax):
        plot_data_and_convolved(axi, dense_history[metric], kernel_size, label=f"dense" if create_labels else None, alpha=0.2, color=f"black")
        for i, (k, v) in enumerate(sparse_histories.items()):
            plot_data_and_convolved(axi, v[metric], kernel_size, label=f"sparse {k}" if create_labels else None, alpha=0.2, color=f"C{i}")
        axi.set_title(metric)
        axi.legend()
        axi.set_xlabel("epoch")
        axi.grid()

def plt_rel_loss(ax, dense_history, sparse_histories, kernel_size=5, create_labels=True):
    metric = "train/loss"
    axi = ax
    axi.hlines(1.0, 0, len(dense_history[metric]), color="black", linestyle="dashed", label="dense" if create_labels else None)
    # plot_data_and_convolved(axi, dense_history[metric], kernel_size, label=f"dense", alpha=0.2, color=f"black")
    for i, (k, v) in enumerate(sparse_histories.items()):
        plot_data = dense_history[metric]/v[metric]
        plot_data_and_convolved(axi, plot_data, kernel_size, label=f"sparse {k}" if create_labels else None, alpha=0.2, color=f"C{i}")
    axi.set_title(metric)
    axi.legend()
    axi.set_xlabel("epoch")
    axi.grid()        


def build_mask(df, **kwargs):
    mask = np.ones(len(df), dtype=bool)
    for k, v in kwargs.items():
        mask = mask & (df[k] == v)        
    return mask

def subselect(df, **kwargs):
    mask = build_mask(df, **kwargs)
    return df[mask]

def get_best_run(df, metric, **kwargs):
    mask = build_mask(df, **kwargs)
    return df[mask].sort_values(metric, ascending=False).iloc[:1]


def main():
    api = wandb.Api()
    # DATASET_NAME = "NMNIST"
    # DATASET_NAME = "SHD"
    DATASET_NAME = "DVSGesture"
    entity, project = "jfinkbeiner", f"{DATASET_NAME}_sweep"  # set to your entity and project 
    runs = api.runs(entity + "/" + project) 

    if DATASET_NAME == "NMNIST":
        KERNEL_SIZE = 5
    elif DATASET_NAME == "DVSGesture":
        KERNEL_SIZE = 5
    elif DATASET_NAME == "SHD":
        KERNEL_SIZE = 9
    else:
        raise ValueError("unknown dataset")
    if not (KERNEL_SIZE % 2 == 1):
        raise ValueError("kernel size must be odd")

    df_list = []
    histories = {}
    for run in runs: 
        if (run.summary.get("_step", 0) < 5):
            continue

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary = run.summary._json_dict

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {
            k: v for k,v in run.config.items()
            if not k.startswith('_')
        }

        df_list.append({"name": run.name, **summary, **config})
        histories[run.name] = run.history()

    runs_df = pd.DataFrame(df_list)
    runs_df = runs_df[runs_df["_step"] > 5]
    # print(runs_df)
    # print(histories[runs_df.name[20]])

    # print(runs_df.columns)
    # print(runs_df["_timestamp"])

    USE_AUG = 0
    runs_arg_dict = {}
    # get unique values for each arg from dense runs
    # RELEVANT_ARGS_DENSE.pop(RELEVANT_ARGS_DENSE.index("use_aug"))
    # runs_arg_dict["use_aug"] = [USE_AUG]
    for arg in RELEVANT_ARGS_DENSE:
        runs_arg_dict[arg] = runs_df[arg].unique().tolist()
    print("all unique args for dense runs:")
    print(runs_arg_dict)



    # generate figures
    fig_comb, ax_comp = plt.subplots(1, 2, sharey=True, sharex=True)
    ax_comp[0].set_ylabel("relative accuracy sparse/dense")
    # ax[0].set_ylim(0.99, 1.02)


    # plot dense and sparse accuracies in one figure
    # left plot is train/acc right plot is test/acc
    fig_abs_acc, ax_abs_acc = plt.subplots(1, 2, sharey=True, sharex=True)
    ax_abs_acc[0].set_ylabel("accuracy")

    # plot dense and sparse losses in one figure
    fig_loss, ax_loss = plt.subplots(1, 1, sharey=True, sharex=True)
    ax_loss.set_ylabel("loss dense/sparse")


    selected_runs = []

    num_plots = 0
    for arg_values in itertools.product(*runs_arg_dict.values()):
        select_kwargs = {arg: arg_values[i] for i, arg in enumerate(runs_arg_dict.keys())}

        # dense_dfs.append(get_best_run(runs_df, "test/acc", **arg_dict))
        # sparse_dfs.append(get_best_run(runs_df, "test/acc", **arg_dict, sparse=True))

        dense_df = subselect(runs_df, use_sparse=False, **select_kwargs)
        sparse_df = subselect(runs_df, use_sparse=True, **select_kwargs)

        sparse_df = sparse_df.sort_values("max_activity", ascending=False)

        # print("\n############ dense run ############")
        # print(dense_df)
        # print("\n############ sparse runs ############")
        # print(sparse_df)

        # if len(dense_df) > 1:
        #     print("WARNING: multiple dense runs found, selecting best one")
        #     dense_df = get_best_run(dense_df, "test/max_acc")

        if len(sparse_df) > 3:
            print("WARNING: multiple sparse runs found, selecting best one")
            # for every value of max_activity filter out best run
            sparse_df = sparse_df.groupby("max_activity").apply(get_best_run, "test/max_acc")

        print("lenghts of dfs:", len(dense_df), len(sparse_df))
        print(sparse_df)
        if not (len(dense_df) == 1 and len(sparse_df) == 3):
            continue


        selected_runs.append(dense_df)
        selected_runs.append(sparse_df)


        print("selected args:")
        print(select_kwargs)


        # sparse_df = get_best_run(runs_df, "test/acc", use_sparse=True, **select_kwargs)
        # assert len(dense_df) == 1

        # for row in sparse_df.itertuples():
        #     print(row)
            # print(row.name, row.max_act, row.max_act_acc)


        dense_history = histories[dense_df.name.iloc[0]]
        sparse_histories = {
            row.max_activity: histories[row.name] for row in sparse_df.itertuples()
        }


        #


        # if select_kwargs["use_aug"] != USE_AUG:
        #     continue
        create_labels = num_plots == 0
        plot_acc_comparison(ax_comp, dense_history, sparse_histories, kernel_size=KERNEL_SIZE, create_labels=create_labels)
        plt_abs_accuracies(ax_abs_acc, dense_history, sparse_histories, kernel_size=KERNEL_SIZE, create_labels=create_labels)
        plt_rel_loss(ax_loss, dense_history, sparse_histories, kernel_size=KERNEL_SIZE, create_labels=create_labels)
        num_plots += 1

    if DATASET_NAME == "NMNIST":
        ax_comp[0].set_ylim(0.985, 1.01)
        ax_abs_acc[0].set_ylim(0.95, 1.005)
    elif DATASET_NAME == "SHD":
        ax_comp[0].set_ylim(0.85, 1.15)
        ax_abs_acc[0].set_ylim(0.55, 1.005)

    for i in range(2):
        ax_comp[i].grid()
        ax_abs_acc[i].grid()
    ax_loss.grid()

    fig_comb.tight_layout()
    fig_abs_acc.tight_layout()
    fig_loss.tight_layout()

    # fig_comb.savefig(f"{DATASET_NAME}_fig_comb.svg")
    # fig_abs_acc.savefig(f"{DATASET_NAME}_fig_abs_acc.svg")
    # fig_loss.savefig(f"{DATASET_NAME}_fig_loss.svg")

    # skwargs_runs_df = pd.concat(selected_runs)
    skwargs_runs_df = runs_df



    # # plot box plots of test accuracies for dense and sparse
    # fig_box, ax_box = plt.subplots(1, 1, sharey=True, sharex=True)
    # ax_box.set_ylabel("test accuracy")
    max_acts = subselect(skwargs_runs_df, use_sparse=True)["max_activity"].unique().tolist()
    max_acts.sort()
    print(max_acts)
    # # generate xticks for [dense, sparse, ...max_acts...]
    # xticks = ["dense"] + ["sparse"] + [f"sparse {max_act}" for max_act in max_acts]
    # ax_box.set_xticks(range(len(xticks)))
    # ax_box.set_xticklabels(xticks)
    # # generate the plot data
    for USE_AUG in [True, False]:
        selected_runs_df = subselect(skwargs_runs_df, use_aug=USE_AUG)
        box_data_dict = {}
        for metric in ["test/max_acc", "train/loss"]:

            box_data = []
            box_data.append(subselect(selected_runs_df, use_sparse=False)[metric].values)
            box_data.append(subselect(selected_runs_df, use_sparse=True)[metric].values)
            for i, max_act in enumerate(max_acts):
                box_data.append(subselect(selected_runs_df, use_sparse=True, max_activity=max_act)[metric].values)
            box_data_dict[metric] = box_data
            # # boxlot with all data points explititly as dot and mean
            # ax_box.boxplot(box_data, showmeans=True, meanline=True, showfliers=False)
        

        sns.set(style="whitegrid")
        # tips = sns.load_dataset("tips")


        for metric in ["test/max_acc", "train/loss"]:
            plt.figure()
            ax = sns.boxplot(data=box_data_dict[metric], showfliers = False) # y="test accuracy",
            ax = sns.swarmplot(data=box_data_dict[metric], color=".25")
            # set xticks
            xticks = ["dense"] + ["sparse"] + [f"sparse\n{max_act}" for max_act in max_acts]
            # ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(xticks)
            ax.set_ylabel(metric)
            ax.set_title(f"use_aug={USE_AUG}")

    
    plt.show()






if __name__ == "__main__":
    
    
    
    
    
    main()

