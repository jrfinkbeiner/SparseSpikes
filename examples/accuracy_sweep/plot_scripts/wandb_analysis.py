import sys
import numpy as np
import pandas as pd 
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
DATASET_NAME = "SHD"
entity, project = "jfinkbeiner", f"{DATASET_NAME}_sweep"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 

# summary_list, config_list, name_list = [], [], []
# histories = []
# for run in runs: 
#     # .summary contains the output keys/values for metrics like accuracy.
#     #  We call ._json_dict to omit large files 
#     summary_list.append(run.summary._json_dict)
#     histories.append(run.history())

#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append(
#         {k: v for k,v in run.config.items()
#          if not k.startswith('_')})

#     # .name is the human-readable name of the run.
#     name_list.append(run.name)

# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
#     })

# print(runs_df)


# print(summary_list[0])


# print(histories[0])
# print(config_list[0])

# runs_df.to_csv("project.csv")


summary_list, config_list, name_list = [], [], []
df_list = []
histories = {}
for run in runs: 
    if (run.summary.get("_step", 0) < 5):
        continue

    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    df_list.append({"name": run.name, **run.summary._json_dict, **run.config})
    histories[run.name] = run.history()

runs_df = pd.DataFrame(df_list)

# print(runs_df)
# print(histories[runs_df.name[20]])




ALL_ARGS = [
    "use_sparse"
    "batchsize"
    "lr"
    "use_lr_scheduler"
    "second_threshold"
    "dataset_name"
    "max_activity"
    "num_hidden_layers"
    "sparse_size_inp"
    "seq_len"
    "num_epochs"
    "batchsize_test"
    "use_wandb"
    "use_bias"
    "use_lsuv"
    "act_rate"
    "ro_type"
    "ro_int"
    "use_thresh_scheduler"
    "thresh_scheduler_mul"
    "use_aug"
    "use_crop"
]

RELEVANT_ARGS_DENSE = [
    # "use_sparse"
    "batchsize"
    # "lr"
    "use_lr_scheduler"
    # "second_threshold"
    # "dataset_name"
    # "max_activity"
    "num_hidden_layers"
    # "sparse_size_inp"
    "seq_len"
    "num_epochs"
    # "batchsize_test"
    # "use_wandb"
    "use_bias"
    "use_lsuv"
    # "act_rate"
    "ro_type"
    # "ro_int"
    # "use_thresh_scheduler"
    # "thresh_scheduler_mul"
    "use_aug"
    "use_crop"
]

RELEVANT_ARGS_SPARSE = [
    *RELEVANT_ARGS_DENSE,
    "use_sparse",
    "sparse_size_inp",
    "second_threshold",
    "use_thresh_scheduler",
    "thresh_scheduler_mul"
]


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


if DATASET_NAME == "MNIST":
    KERNEL_SIZE = 5
    select_kwargs = dict(batchsize=256, use_aug=True, use_lr_scheduler=True, ro_type="linear_ro", num_hidden_layers=3, epoch=49, seq_len=250)
# elif DATASET_NAME == "DVSGesture":
    # KERNEL_SIZE = 4
#     select_kwargs = dict(batchsize=256, use_aug=True, use_lr_scheduler=True, ro_type="linear_ro", num_hidden_layers=3, epoch=49, seq_len=250)
elif DATASET_NAME == "SHD":
    KERNEL_SIZE = 9
    select_kwargs = dict(batchsize=256, use_aug=False, use_lr_scheduler=True, ro_type="linear_ro", num_hidden_layers=4, seq_len=500)
else:
    raise ValueError("unknown dataset")
if not (KERNEL_SIZE % 2 == 1):
    raise ValueError("kernel size must be odd")
dense_df = subselect(runs_df, use_sparse=False, **select_kwargs)
sparse_df = subselect(runs_df, use_sparse=True, **select_kwargs)


print("\n############ dense run ############")
print(dense_df)
print("\n############ sparse runs ############")
print(sparse_df)

if len(dense_df) != 1:
    print("WARNING: multiple dense runs found, selecting best one")
    dense_df = get_best_run(dense_df, "test/max_acc")

print(dense_df)

# sparse_df = get_best_run(runs_df, "test/acc", use_sparse=True, **select_kwargs)
# assert len(dense_df) == 1

# for row in sparse_df.itertuples():
#     print(row)
    # print(row.name, row.max_act, row.max_act_acc)



dense_history = histories[dense_df.name.iloc[0]]
sparse_histories = {
    row.max_activity: histories[row.name] for row in sparse_df.itertuples()
}


def plot_data_and_convolved(ax, data, kernel_size, *, label, alpha=0.2, **kwargs):
    ax.plot(data, alpha=alpha, **kwargs)
    ax.plot(np.arange((kernel_size)//2, len(data)-kernel_size//2), np.convolve(data, np.ones((kernel_size,))/kernel_size, mode='valid'), label=label, **kwargs)


fig_comb, ax_comp = plt.subplots(1, 2, sharey=True, sharex=True)
for metric, axi in zip(["train/acc", "test/acc"], ax_comp):
    axi.hlines(1.0, 0, len(dense_history[metric]), color="black", linestyle="dashed", label="dense")
    for i, (k, v) in enumerate(sparse_histories.items()):
        plot_data = v[metric]/dense_history[metric]
        plot_data_and_convolved(axi, plot_data, KERNEL_SIZE, label=f"sparse {k}", alpha=0.2, color=f"C{i}")
        # axi.plot(plot_data, alpha=0.2, color=f"C{i}")
        # axi.plot(np.arange((KERNEL_SIZE)//2, len(plot_data)-KERNEL_SIZE//2), np.convolve(plot_data, np.ones((KERNEL_SIZE,))/KERNEL_SIZE, mode='valid'), color=f"C{i}", label=k)
    axi.set_title(metric)
    axi.legend()
    axi.set_xlabel("epoch")
ax_comp[0].set_ylabel("test accuracy sparse/dense")
# ax[0].set_ylim(0.99, 1.02)
fig_comb.tight_layout()


# plot dense and sparse accuracies in one figure
# left plot is train/acc right plot is test/acc
fig_abs_acc, ax_abs_acc = plt.subplots(1, 2, sharey=True, sharex=True)
for metric, axi in zip(["train/acc", "test/acc"], ax_abs_acc):
    plot_data_and_convolved(axi, dense_history[metric], KERNEL_SIZE, label=f"dense", alpha=0.2, color=f"black")
    for i, (k, v) in enumerate(sparse_histories.items()):
        # print(metric, i, k, v[metric].values.shape)
        plot_data_and_convolved(axi, v[metric], KERNEL_SIZE, label=f"sparse {k}", alpha=0.2, color=f"C{i}")
    axi.set_title(metric)
    axi.legend()
    axi.set_xlabel("epoch")
    axi.grid()
ax_abs_acc[0].set_ylabel("accuracy")
fig_abs_acc.tight_layout()

plt.show()

