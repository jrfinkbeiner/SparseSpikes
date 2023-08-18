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
    x = np.linspace(0.0, 1.0, len(data))
    ax.plot(x, data, alpha=alpha, **kwargs)
    x_conv = np.arange((kernel_size)//2, len(data)-kernel_size//2) / len(data)
    ax.plot(x_conv, np.convolve(data, np.ones((kernel_size,))/kernel_size, mode='valid'), label=label, **kwargs)


def plot_acc_comparison(ax, dense_history, sparse_histories, kernel_size=5, create_labels=True):
    for metric, axi in zip(["train/acc", "test/acc"], ax):
        axi.hlines(1.0, 0, len(dense_history[metric]), color="black", linestyle="dashed", label="dense" if create_labels else None)
        for i, (k, v) in enumerate(sparse_histories.items()):
            plot_data = v[metric]/dense_history[metric]
            plot_data_and_convolved(axi, plot_data, kernel_size, label=f"sparse {k}" if create_labels else None, alpha=0.2, color=f"C{i}")
        axi.set_title(metric)
        axi.legend()
        axi.set_xlabel("epoch")


def plt_abs_accuracies(ax, metric, dense_history, sparse_histories, kernel_size=5, create_labels=True):
    plot_data_and_convolved(ax, dense_history[metric], kernel_size, label=f"dense" if create_labels else None, alpha=0.2, color=f"black")
    for i, (k, v) in enumerate(sparse_histories.items()):
        plot_data_and_convolved(ax, v[metric], kernel_size, label=f"sparse {k}" if create_labels else None, alpha=0.2, color=f"C{i}")
    ax.set_title(metric)
    ax.legend()
    ax.set_xlabel("epoch")
    ax.grid()

def plt_abs_accuracies_with_std(ax, metric, dense_history, sparse_histories, dense_histories_std, sparse_histories_std, kernel_size=5, create_labels=True):
   # plot_data_and_convolved(ax, dense_history[metric], kernel_size, label=f"dense" if create_labels else None, alpha=0.2, color=f"black")
    # plot mean and std of dense
    # ax.plot(dense_history[metric], label=f"dense" if create_labels else None, alpha=0.8, color=f"black")
    plot_data_and_convolved(ax, dense_history[metric], kernel_size, label=f"dense" if create_labels else None, alpha=0.2, color=f"black")
    ax.fill_between(np.linspace(0.0, 1.0, len(dense_history[metric])), dense_history[metric]-dense_histories_std[metric], dense_history[metric]+dense_histories_std[metric], alpha=0.1, color=f"black")

    for i, (k, v) in enumerate(sparse_histories.items()):
        plot_data_and_convolved(ax, v[metric], kernel_size, label=f"{int(k*100):2}"+r"\% act" if create_labels else None, alpha=0.2, color=f"C{i}")
        # plot mean and std of sparse
        # ax.plot(v[metric], label=f"sparse {k}" if create_labels else None, alpha=0.8, color=f"C{i}")
        ax.fill_between(np.linspace(0.0, 1.0, len(v[metric])), v[metric]-sparse_histories_std[k][metric], v[metric]+sparse_histories_std[k][metric], alpha=0.1, color=f"C{i}")
    ax.grid()


def plt_rel_metric(ax, metric, dense_history, sparse_histories, kernel_size=5, create_labels=True):
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

def get_dataframes(runs):

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
    runs_df = runs_df[runs_df["use_lr_scheduler"] == 1]

    USE_AUG = 0
    runs_arg_dict = {}
    # get unique values for each arg from dense runs
    # RELEVANT_ARGS_DENSE.pop(RELEVANT_ARGS_DENSE.index("use_aug"))
    # runs_arg_dict["use_aug"] = [USE_AUG]
    for arg in RELEVANT_ARGS_DENSE:
        runs_arg_dict[arg] = runs_df[arg].unique().tolist()
    print("all unique args for dense runs:")
    print(runs_arg_dict)


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

        if len(dense_df) > 1:
            print("WARNING: multiple dense runs found, selecting best one")
            dense_df = get_best_run(dense_df, "test/max_acc")

        if len(sparse_df) > 3:
            print("WARNING: multiple sparse runs found, selecting best one")
            # for every value of max_activity filter out best run
            sparse_df = sparse_df.groupby("max_activity").apply(get_best_run, "test/max_acc")

        print("lenghts of dfs:", len(dense_df), len(sparse_df))
        if not (len(dense_df) == 1 and len(sparse_df) == 3):
            continue
        no_failed_run = np.all(dense_df["epoch"].values[0] == sparse_df["epoch"].values)
        if not no_failed_run:
            continue


        selected_runs.append(dense_df)
        selected_runs.append(sparse_df)    

    return pd.concat(selected_runs), dense_df, sparse_df, histories


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def create_final_acc_box_plot(ax, skwargs_runs_df, max_acts, metric="test/max_acc", size=3, marker="o", **kwargs):
    for USE_AUG in [None]:
        if USE_AUG is None:
            selected_runs_df = skwargs_runs_df
        else:
            selected_runs_df = subselect(skwargs_runs_df, use_aug=USE_AUG)
        # box_data_dict = {}


        box_data = []
        box_data.append(subselect(selected_runs_df, use_sparse=False)[metric].values*100)
        box_data.append(subselect(selected_runs_df, use_sparse=True)[metric].values*100)
        for i, max_act in enumerate(max_acts):
            box_data.append(subselect(selected_runs_df, use_sparse=True, max_activity=max_act)[metric].values*100)
        # box_data_dict[metric] = box_data
        # # boxlot with all data points explititly as dot and mean
        # ax_box.boxplot(box_data, showmeans=True, meanline=True, showfliers=False)
        



        print("box_data")
        print(box_data)

        # create dataframe from numpy array
        dfs = [pd.DataFrame(data={"Variable": np.full(len(box_data[i]), i), "Accuracy": box_data[i]}) for i in range(len(box_data))]
        melted_df = pd.concat(dfs)
        # print("dfs")
        # print(dfs)

        # random_df = pd.DataFrame(data={f"{i}": box_data[i] for i in range(len(box_data))})
        
        # print("random_df")
        # print(random_df)

        # xvalues = ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5"]
        palette = ['k', f'C{4}', f'C{0}', f'C{1}', f'C{2}']
        print("palette", palette)
        # apply alpha factor to palette colors
        import matplotlib as mpl
        # palette = [mpl.colors.to_hex(mpl.colors.to_rgba(palette[0], 0.1))] + [mpl.colors.to_hex(mpl.colors.to_rgba(fc, 0.0)) for fc in palette[1:]]
        palette = [lighten_color(palette[0], 0.3)] + [lighten_color(fc, 0.82) for fc in palette[1:]]

        print("palette", palette)

        # melted_df = random_df.set_axis(xvalues, axis=1).melt(var_name='Variable', value_name='Accuracy')

        print("melted_df")
        print(melted_df)

        sns.boxplot(data=melted_df, x='Variable', y='Accuracy', palette=palette,
                    width=0.7, ax=ax, showfliers = False, linewidth=1)
        # ax.legend_.remove()  # remove the legend, as the information is already present in the x labels
        ax.set_ylabel('')
        ax.set_xlabel("")

        for color, patch, in zip(["k", "C5", "C0", "C1", "C2"], ax.artists):
            # fc = patch.get_facecolor()
            fc = color
            patch.set_facecolor(plt.colors.to_rgba(fc, 0.3))

        # create x positions for data points
        y_pos = np.concatenate(box_data)
        x_pos = np.repeat(np.arange(1, len(box_data)+1), [arr.size for arr in box_data])

        # give label only once for all arrays in box_data_dict[metric]
        # sns.swarmplot(ax=ax, data=box_data_dict[metric], color=".25", size=size, marker=marker, **kwargs, legend="brief", ) # y="test accuracy",
        sns.swarmplot(ax=ax, x=x_pos, y=y_pos, color=".25", size=size, marker=marker, **kwargs) # y="test accuracy",
        # sns.swarmplot(ax=ax, data=box_data_dict[metric].melt(id_vars="model"), x="model", y="value",hue="variable", color=".25", size=size, marker=marker, **kwargs, legend="brief") # y="test accuracy",


        # set xticks
        xticks = ["Dense"] + ["Sparse"] + [f"{int(max_act*100)}"+r"\%" for max_act in max_acts]
        xticks[-2] = xticks[-2] + "\n" + r"Maximal Activity" 

        # ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks)
        # ax.set_ylabel(metric)
        # ax.set_title(f"use_aug={USE_AUG}")


def broken_barh(ax, ax2, color='k', d = .015):
    #     # zoom-in / limit the view to different portions of the data
    # ax.set_ylim(.78, 1.)  # outliers only
    # ax2.set_ylim(0, .22)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('none') # don't put tick labels at the top
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

      # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color=color, clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'


def accuracy_plot(skwargs_runs_dfs, ax_acc, metric, axes_and_ticks_color):
    

    dataset_to_marker = {
        "NMNIST": "o",
        "DVSGesture": "X",
        "SHD": "d",
    }

    marker_to_size = {
        "o": 3,
        "X": 4,
        "*": 5,
        "d": 4,
    }

    for idat, DATASET_NAME in enumerate(["NMNIST", "DVSGesture", "SHD"]):

        skwargs_runs_df = skwargs_runs_dfs[DATASET_NAME]

        max_acts = subselect(skwargs_runs_df, use_sparse=True)["max_activity"].unique().tolist()
        max_acts.sort(reverse=True)


        marker = dataset_to_marker[DATASET_NAME]
        create_final_acc_box_plot(ax_acc[idat], skwargs_runs_df, max_acts, metric=metric, marker=marker, size=marker_to_size[marker])
        create_final_acc_box_plot(ax_acc[-1], skwargs_runs_df, max_acts, metric=metric, marker=marker, size=marker_to_size[marker]) #, legend="brief") # TODO uncomment

        broken_barh(ax_acc[idat], ax_acc[idat+1], color=axes_and_ticks_color, d=0.015)


    # set y lim for all plots
    if metric=="test/max_acc":
        mul = 100
        ax_acc[ 0].set_ylim(0.983*mul, 0.9925*mul)
        ax_acc[ 1].set_ylim(0.885*mul, 0.94*mul)
        ax_acc[ 2].set_ylim(0.65*mul, 0.81*mul)
        ax_acc[-1].set_ylim(0.0*mul, 0.61*mul)

    # create dashed vline at 2.5
    for axi in ax_acc:
        axi.vlines(1.5, *axi.get_ylim(), color="black", linestyle="dashed", alpha=0.5)


    for axi in ax_acc:
        axi.grid(True, axis="y", alpha=0.3)

        # ax_acc.set_ylim(0.0, 1.05)


    # manually set marker labels to ["NMNIST", "DVSGesture", "SHD"] with the corresponding markers from dataset_to_marker

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], marker=dataset_to_marker[DATASET_NAME], color='w', label=DATASET_NAME,
                            markerfacecolor='k', markersize=5) for DATASET_NAME in ["NMNIST", "DVSGesture", "SHD"]]
    # marker_to_size[dataset_to_marker[DATASET_NAME]]
    ax_acc[-1].legend(frameon=True, handles=legend_elements)


    # ax_acc[-1].legend(frameon=False)




def histroy_plot(ax_hist, DATASET_NAME, skwargs_runs_dfs, dense_dfs, sparse_dfs, all_histories):

    # DATASET_NAME = "SHD"
    # DATASET_NAME = "DVSGesture"

    if DATASET_NAME == "NMNIST":
        KERNEL_SIZE = 5
    elif DATASET_NAME == "DVSGesture":
        KERNEL_SIZE = 5
    elif DATASET_NAME == "SHD":
        KERNEL_SIZE = 9


    histories = all_histories[DATASET_NAME]
    dense_df = dense_dfs[DATASET_NAME]
    sparse_df = sparse_dfs[DATASET_NAME]
    skwargs_runs_df = skwargs_runs_dfs[DATASET_NAME]


    print("dense_df")
    print(dense_df.shape)
    print(dense_df)
    print("sparse_df")
    print(sparse_df.shape)
    print(sparse_df)
    print("skwargs_runs_df")
    print(skwargs_runs_df.shape)
    print(skwargs_runs_df)


    num_plots = 0
    assert skwargs_runs_df.shape[0] % 4 == 0
    num_runs = skwargs_runs_df.shape[0] // 4
    for i in range(num_runs):
        dense_run = skwargs_runs_df[4*i:4*i+1]
        sparse_runs = skwargs_runs_df[4*i+1:4*i+4]

        dense_history = histories[dense_run.name.iloc[0]]
        sparse_histories = {
            row.max_activity: histories[row.name] for row in sparse_runs.itertuples()
        }

        # if select_kwargs["use_aug"] != USE_AUG:
        #     continue
        create_labels = num_plots == 0
        # plot_acc_comparison(ax_comp, dense_history, sparse_histories, kernel_size=KERNEL_SIZE, create_labels=create_labels)
        
        
        # plt_rel_metric(ax_hist[0], "train/loss", dense_history, sparse_histories, kernel_size=KERNEL_SIZE, create_labels=create_labels)
        # plt_rel_metric(ax_hist[1], "test/acc", dense_history, sparse_histories, kernel_size=KERNEL_SIZE, create_labels=create_labels)

        plt_abs_accuracies(ax_hist[0], "train/loss", dense_history, sparse_histories, kernel_size=KERNEL_SIZE, create_labels=create_labels)
        plt_abs_accuracies(ax_hist[1], "test/acc", dense_history, sparse_histories, kernel_size=KERNEL_SIZE, create_labels=create_labels)
        
        
        ax_hist[0].set_yscale("log")
        num_plots += 1


def mean_histroy_plot(ax_hist, DATASET_NAME, skwargs_runs_dfs, dense_dfs, sparse_dfs, all_histories):

    # DATASET_NAME = "SHD"
    # DATASET_NAME = "DVSGesture"

    if DATASET_NAME == "NMNIST":
        KERNEL_SIZE = 5
    elif DATASET_NAME == "DVSGesture":
        KERNEL_SIZE = 5
    elif DATASET_NAME == "SHD":
        KERNEL_SIZE = 9


    histories = all_histories[DATASET_NAME]
    dense_df = dense_dfs[DATASET_NAME]
    sparse_df = sparse_dfs[DATASET_NAME]
    skwargs_runs_df = skwargs_runs_dfs[DATASET_NAME]


    print("dense_df")
    print(dense_df.shape)
    print(dense_df)
    print("sparse_df")
    print(sparse_df.shape)
    print(sparse_df)
    print("skwargs_runs_df")
    print(skwargs_runs_df.shape)
    print(skwargs_runs_df)




    all_dense_histories = []
    all_sparse_histories = {max_act: [] for max_act in sparse_df.max_activity.unique()}
    for i in range(skwargs_runs_df.shape[0]):
        run = skwargs_runs_df[i:i+1]
        print("run.name", run.name.iloc[0])
        print("run", run)
        history = histories[run.name.iloc[0]]
        if run.use_sparse.iloc[0]:
            all_sparse_histories[run.max_activity.iloc[0]].append(history)
        else:
            all_dense_histories.append(history)


    def combine_histories(histories, metric):
        selected_histories = [history[metric] for history in histories]
        max_len = max([len(history[metric]) for history in histories])
        selected_histories_arr = np.full((len(selected_histories), max_len), np.nan)
        for i, history in enumerate(selected_histories):
            selected_histories_arr[i, :len(history)] = history
            selected_histories_arr[i, len(history):] = np.mean(history[-KERNEL_SIZE:])

        return selected_histories_arr

    # calculate mean and std over histories
    dense_history_mean = {}
    dense_history_std = {}
    for metric in all_dense_histories[0].keys():
        selected_histories_arr = combine_histories(all_dense_histories, metric)
        if "acc" in metric:
            selected_histories_arr = selected_histories_arr * 100
        dense_history_mean[metric] = np.nanmean(selected_histories_arr, axis=0)
        dense_history_std[metric] = np.nanstd(selected_histories_arr, axis=0)

    sparse_histories_mean = {}
    sparse_histories_std = {}
    for max_act, histories in all_sparse_histories.items():
        sparse_histories_mean[max_act] = {}
        sparse_histories_std[max_act] = {}
        for metric in histories[0].keys():
            selected_histories_arr = combine_histories(histories, metric)
            if "acc" in metric:
                selected_histories_arr = selected_histories_arr * 100
            sparse_histories_mean[max_act][metric] = np.nanmean(selected_histories_arr, axis=0)
            sparse_histories_std[max_act][metric] = np.nanstd(selected_histories_arr, axis=0)

    # plt_abs_accuracies(ax_hist[0], "train/loss", dense_history_mean, sparse_histories_mean, kernel_size=KERNEL_SIZE, create_labels=True)
    # plt_abs_accuracies(ax_hist[1], "test/acc", dense_history_mean, sparse_histories_mean, kernel_size=KERNEL_SIZE, create_labels=True)
    

    plt_abs_accuracies_with_std(ax_hist[0], "train/loss", dense_history_mean, sparse_histories_mean, dense_history_std, sparse_histories_std, kernel_size=KERNEL_SIZE, create_labels=True)
    plt_abs_accuracies_with_std(ax_hist[1], "test/acc", dense_history_mean, sparse_histories_mean, dense_history_std, sparse_histories_std, kernel_size=KERNEL_SIZE, create_labels=True)


def main(api, entity):    

    skwargs_runs_dfs, dense_dfs, sparse_dfs, all_histories = {}, {}, {}, {}


    for idat, DATASET_NAME in enumerate(["NMNIST", "DVSGesture", "SHD"]):

        project = f"{DATASET_NAME}_sweep"
        runs = api.runs(entity + "/" + project) 

        skwargs_runs_dfs[DATASET_NAME], dense_dfs[DATASET_NAME], sparse_dfs[DATASET_NAME], all_histories[DATASET_NAME] = get_dataframes(runs)


    # from tueplots import bundles
    # bundle = bundles.neurips2023()
    bundle = {'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': '\\renewcommand{\\rmdefault}{ptm}\\renewcommand{\\sfdefault}{phv}', 'figure.figsize': (5.5, 3.399186938124422), 'figure.constrained_layout.use': True, 'figure.autolayout': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.015, 'font.size': 10, 'legend.fontsize': 7.5, 'axes.titlesize': 10} #, 'axes.labelsize': 9, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, }
    # Plug any of those into either the rcParams or into an rc_context:
    plt.rcParams.update(bundle)
    
    axes_and_ticks_color = '#'+'b'*6
    plt.rc('axes',edgecolor=axes_and_ticks_color)
    plt.rcParams['xtick.color'] = axes_and_ticks_color
    plt.rcParams['ytick.color'] = axes_and_ticks_color
    plt.rcParams['xtick.labelcolor'] = "k"
    plt.rcParams['ytick.labelcolor'] = "k"
    

    fig = plt.figure(figsize=(7,3), tight_layout=True)
    gs1 = fig.add_gridspec(nrows=4, ncols=1, left=0.055, right=0.35, bottom=0.15, top=0.928, hspace=0.06)
    ax_acc = gs1.subplots(sharex=True, sharey=False)
    gs2 = fig.add_gridspec(nrows=3, ncols=2, left=0.40, right=0.98, bottom=0.15, top=0.928, hspace=0.1)
    ax_hist = gs2.subplots(sharex=True, sharey=False)

    # with sns.axes_style("whitegrid"):
    # for i,metric in enumerate(["test/max_acc", "train/loss"]):
    for i,metric in enumerate(["test/max_acc"]):
        accuracy_plot(skwargs_runs_dfs, ax_acc, metric, axes_and_ticks_color)

    ax_acc[0].set_title("Best Test Accuracy [\%]")


    for i,dataset_name in enumerate(["NMNIST", "DVSGesture", "SHD"]):
        mean_histroy_plot(ax_hist[i], dataset_name, skwargs_runs_dfs, dense_dfs, sparse_dfs, all_histories)

    use_y_log = True

    if use_y_log:
        ax_hist[0, 0].set_yscale("log")
        ax_hist[1, 0].set_yscale("log")
        ax_hist[2, 0].set_yscale("log")
        ax_hist[0, 0].set_ylim(80, 1300)
        ax_hist[1, 0].set_ylim(500, 10200)
        ax_hist[2, 0].set_ylim(90, 6100)
    else:
        ax_hist[0, 0].set_ylim(-100, 1300)
        ax_hist[1, 0].set_ylim(-500, 10100)
        ax_hist[2, 0].set_ylim(-200, 6100)
    ax_hist[0, 1].set_ylim(0.97*100, 0.991*100)
    ax_hist[1, 1].set_ylim(0.6*100, 0.955*100)
    ax_hist[2, 1].set_ylim(0.3*100, 0.8*100)


    for axi in ax_hist.flatten():
        axi.set_xlim(0.0, 1.0)

    # restirct x ticks to [0.0, 0.5, 1.0]
    for axi in ax_hist[0, :]:
        axi.set_xticks([0.0, 0.5, 1.0])

    ax_hist[0, 0].text(0.8, 800, "NMNIST", horizontalalignment='center', size=9.5)
    ax_hist[1, 0].text(0.8, 6000, "DVSGesture", horizontalalignment='center', size=9.5)
    ax_hist[2, 0].text(0.8, 2800, "SHD", horizontalalignment='center', size=9.5)
    ax_hist[0, 1].text(0.26, 97.15, "NMNIST", size=9.5)
    ax_hist[1, 1].text(0.73, 63.5, "DVSGesture", horizontalalignment='center', size=9.5)
    ax_hist[2, 1].text(0.73, 33.5, "SHD", horizontalalignment='center', size=9.5)



    ax_hist[0, 0].set_title("Training Loss")
    ax_hist[0, 1].set_title(r"Test Accuracy [\%]")
    ax_hist[0, 1].legend(frameon=False, loc=(0.6,-0.05))
    # Place x label to the right
    ax_hist[2, 0].set_xlabel(r"Training Process [Epoch / Total Epochs]", ha="right", x=1.67)
    # ax_hist[2, 1].set_xlabel("training process [epoch / total epochs]")


    # fig.savefig(f"./figures/accuracy_conversion.pdf")
    plt.show()


if __name__ == "__main__":
    # TODO set wandb api to your personal choice
    wandb_entity = ... # TODO set
    api = wandb.Api(overrides={"base_url": ...}) # TODO set
    main(api)