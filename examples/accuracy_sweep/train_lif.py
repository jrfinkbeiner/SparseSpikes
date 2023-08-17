import sys
from typing import Optional
import functools as ft
import pprint
import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax



from sparsespikes.jax_interface import SparseSpikeVector

from nmnist_util import create_gener #, get_dataloader, get_tonic_prototyping_dataloader
from architecture import init_network_weights, init_network_states, create_one_hot, calc_loss_batch, lif_network, sum_and_crossentropy
from util import calc_mean_activity


def get_update_fn(optimizer, loss_fn, sparse_sizes=None):
    def update_fn(weights, thresholds, alphas, initial_state, inp_spikes, labels, opt_state):
        """
        This function calculates the weight updates of the model by computing the gradients.
        """
        (loss, aux), grads = jax.value_and_grad(ft.partial(calc_loss_batch, sparse_sizes=sparse_sizes), has_aux=True)(weights, thresholds, alphas, initial_state, inp_spikes, labels, loss_fn)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        updated_weights = optax.apply_updates(weights, updates)    
        return updated_weights, opt_state, loss, grads, aux
    return update_fn

# # @ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
def calc_accuracy(weights, thresholds, alphas, initial_state, inp_spikes, target, calc_accuracy_from_output_fn, *, sparse_sizes=None):
    final_state, out_spikes = lif_network(weights, thresholds, alphas, initial_state, inp_spikes, sparse_sizes=sparse_sizes)
    return calc_accuracy_from_output_fn(out_spikes[-1], target)

def calc_accuracy_from_output_sum_spike(out, target, *, normalized=True):
    pred = out.sum(axis=0).argmax(axis=-1)
    if normalized:
        return (pred==target).mean()
    else:
        return (pred==target).sum()

def calc_pred_from_linear_readout(out, *, num_classes):
    preds_per_timestep = jnp.argmax(out, axis=-1)
    vals, counts = jnp.unique(preds_per_timestep, return_counts=True, size=num_classes)
    return vals[counts.argmax(axis=0)]

def calc_accuracy_from_linear_ro(out, target, *, normalized=True, num_classes):
    preds = jax.vmap(ft.partial(calc_pred_from_linear_readout, num_classes=num_classes), in_axes=(1,))(out)
    if normalized:
        return (preds==target).mean()
    else:
        return (preds==target).sum()


def jax_create_batch(np_batch, num_classes, num_neurons: Optional[int] = None):
    sparse_data: bool = "inp_spike_ids" in np_batch
    if sparse_data:
        assert num_neurons is not None
        spike_ids = np_batch["inp_spike_ids"].transpose((1,0,2))
        num_spikes = np.repeat(np_batch["num_inp_spikes"].transpose((1,0))[:, None, :], 2, axis=1)
        input_spikes = SparseSpikeVector(spike_ids=spike_ids, num_spikes=num_spikes, num_neurons=num_neurons)
    else:
        # transpose to switch time and batch axis
        input_spikes = jnp.asarray(np_batch["inp_spikes"].transpose((1,0,2)))
    targets = create_one_hot(np_batch["targets"], num_classes)
    return input_spikes, targets

def threshold_scheduler(mul_fac, thresholds, target):
    return (target-thresholds)*mul_fac+thresholds

def init_LSUV_actrate(act_rate, threshold=0., var=1.0):
    from scipy.stats import norm
    import scipy.optimize
    return scipy.optimize.fmin(lambda loc: (act_rate-(1-norm.cdf(threshold,loc,var)))**2, x0=0.)[0]

def main(args):
    use_wandb = args.use_wandb
    if use_wandb:
        import wandb
        # wandb.init(project=f"{args.dataset_name}_threshold_sweep", config=args) # TODO save the layer sizes!!! and lsuv init params
        wandb.init(project=f"{args.dataset_name}_sweep", config=args) # TODO save the layer sizes!!! and lsuv init params

    DATASET_NAME = args.dataset_name
    ROOT_PATH = args.root_path_data
    SEQ_LEN = args.seq_len
    BATCHSIZE = args.batchsize
    BATCHSIZE_TEST = args.batchsize_test
    SPARSE_SIZE_INP = args.sparse_size_inp
    NUM_EPOCHS = args.num_epochs
    USE_SPARSE = bool(args.use_sparse)
    MAX_ACTIVITY = args.max_activity
    SECOND_THRESHOLD = args.second_threshold
    LEARNING_RATE = args.lr
    USE_BIAS = bool(args.use_bias)
    USE_LSUV = args.use_lsuv
    ACT_RATE = args.act_rate
    RO_TYPE = args.ro_type
    RO_INT = args.ro_int
    

    print("DATASET_NAME:", DATASET_NAME)
    print("SEQ_LEN:", SEQ_LEN)
    print("BATCHSIZE:", BATCHSIZE)
    print("BATCHSIZE_TEST:", BATCHSIZE_TEST)
    print("SPARSE_SIZE_INP:", SPARSE_SIZE_INP)
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("USE_SPARSE:", USE_SPARSE)
    print("MAX_ACTIVITY:", MAX_ACTIVITY)
    print("SECOND_THRESHOLD:", SECOND_THRESHOLD)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("USE_BIAS:", USE_BIAS)
    print("USE_LSUV:", USE_LSUV)
    print("ACT_RATE:", ACT_RATE)
    print("RO_TYPE:", RO_TYPE)
    print("RO_INT:", RO_INT)

    SEED = 42
    rng = np.random.default_rng(SEED)
    key = jrandom.PRNGKey(rng.integers(999999))

    DATASET_TO_IMAGE_DIMS = {
        "NMNIST": (32, 32, 2),
        "DVSGesture":  (48, 48, 2),
        "SHD": (700, 1, 1),
    }
    IMAGE_DIMS = DATASET_TO_IMAGE_DIMS[DATASET_NAME]
    DATASET_NUM_CLASSES = {
        "NMNIST": 10,
        "DVSGesture": 11,
        "SHD": 20,
    }
    NUM_CLASSES = DATASET_NUM_CLASSES[DATASET_NAME]

    if args.num_hidden_layers==2:
        DENSE_SIZES = [np.prod(IMAGE_DIMS), 512, 128, NUM_CLASSES]
    elif args.num_hidden_layers==3:
        DENSE_SIZES = [np.prod(IMAGE_DIMS), 1024, 512, 128, NUM_CLASSES]
    elif args.num_hidden_layers==4:
        DENSE_SIZES = [np.prod(IMAGE_DIMS), 1472, 1024, 512, 128, NUM_CLASSES]
    elif args.num_hidden_layers==5:
        DENSE_SIZES = [np.prod(IMAGE_DIMS), 1472, 1024, 1024, 512, 128, NUM_CLASSES]
    elif args.num_hidden_layers==6:
        DENSE_SIZES = [np.prod(IMAGE_DIMS), 1472, 1024, 1024, 1024, 512, 128, NUM_CLASSES]

    SPARSE_SIZES = [SPARSE_SIZE_INP, *[int(max((MAX_ACTIVITY*dense_size//2)*2, 2)) for dense_size in DENSE_SIZES[1:]]]

    NUM_HIDDEN_LAYERS = len(DENSE_SIZES)-2

    alphas = [0.95]*(NUM_HIDDEN_LAYERS+1)
    if args.use_thresh_scheduler:
        thresholds = jnp.asarray([1.0, -100], dtype=jnp.float32)
        thresholds_target = jnp.asarray([1.0, SECOND_THRESHOLD], dtype=jnp.float32)
    else:
        thresholds = jnp.asarray([1.0, SECOND_THRESHOLD], dtype=jnp.float32)


    INP_DIM = SPARSE_SIZES[0] if USE_SPARSE else DENSE_SIZES[0]
    gen_train, num_samples = create_gener(rng, DATASET_NAME, ROOT_PATH, USE_SPARSE, seq_len=SEQ_LEN, sparse_size=INP_DIM, dataset_split="train", batchsize=BATCHSIZE, shuffle=True, use_multiprocessing=True, use_aug=bool(args.use_aug), use_crop=args.use_crop) #, num_samples=num_samples)
    gen_test, num_samples_test = create_gener(rng, DATASET_NAME, ROOT_PATH, USE_SPARSE, seq_len=SEQ_LEN, sparse_size=INP_DIM, dataset_split="test", batchsize=BATCHSIZE_TEST, shuffle=False, use_multiprocessing=True, use_aug=False, use_crop=args.use_crop)

    NUM_BATCHES = num_samples//BATCHSIZE
    print("NUM_SAMPLES TRAIN:", num_samples)
    print("NUM_SAMPLES_TEST:", num_samples_test)

    key, params_key = jrandom.split(key)
    params = init_network_weights(params_key, DENSE_SIZES, True, dtype=jnp.float32)
    state_dims_dict = {
        "spike_sum": [*DENSE_SIZES[1:], DENSE_SIZES[-1]],
        "linear_ro": DENSE_SIZES[1:-1],
    }
    init_states_fn = ft.partial(init_network_states, state_dims=state_dims_dict[RO_TYPE])

    print(jax.tree_map(lambda x: x.dtype, params))
    # sys.exit()

    if USE_LSUV:
        from lsuv import lsuv
        surr_grad_fn = lambda x: 1./(5*jnp.abs(x)+1.) # TODO why this, not better with other ?
        x0 = np.linspace(-5,5, 1000)
        delta = x0[np.where(surr_grad_fn(x0)>.2)][-1]-x0[np.where(surr_grad_fn(x0)>.2)][0]
        tgt_std = delta/2
        tgt_var = tgt_std**2
        tgt_mu = init_LSUV_actrate(ACT_RATE, threshold=thresholds[0].item(), var=tgt_var)

        print(f"LSUV: tgt_mu={tgt_mu}, tgt_var={tgt_var}, delta={delta}, tgt_std={tgt_std}")

        init_key, key = jrandom.split(key, 2)
        init_states = init_states_fn(BATCHSIZE)
        # call_fn_lsuv = ft.partial(lif_network, thresholds=thresholds, alphas=alphas)
        lsuv_data = jax_create_batch(next(iter(gen_train())), NUM_CLASSES, num_neurons=np.prod(IMAGE_DIMS))[0]
        spike_layer_ids = tuple(range(len(params))) if RO_TYPE=="spike_sum" else tuple(range(len(params)-2)) # -1 for readout layer, -1 for leaky integrate (non fire layer)
        params = lsuv(ft.partial(lif_network, sparse_sizes=SPARSE_SIZES), init_states, params, (thresholds, alphas), spike_layer_ids, lsuv_data, init_key, var_tol=0.1, mean_tol=0.1, tgt_mean=tgt_mu, tgt_var=tgt_var, max_iters=500)

    if args.use_lr_scheduler:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=LEARNING_RATE/5,
            peak_value=LEARNING_RATE,
            warmup_steps=10,
            decay_steps=int(NUM_EPOCHS * NUM_BATCHES),
            end_value=2e-2*LEARNING_RATE
            )
    else:
        lr = LEARNING_RATE
    opt = optax.adamw(lr)



    loss_fn_dict = {
        "spike_sum": ft.partial(sum_and_crossentropy, sum_first=True),
        "linear_ro": ft.partial(sum_and_crossentropy, sum_first=False),
    }

    update_fn = get_update_fn(opt, loss_fn_dict[RO_TYPE], SPARSE_SIZES)

    opt_state = opt.init(params)

    calc_accuracy_from_output_fn_dict = {
        "spike_sum": calc_accuracy_from_output_sum_spike,
        "linear_ro": ft.partial(calc_accuracy_from_linear_ro, num_classes=NUM_CLASSES),
    }
    calc_accuracy_from_output_fn = calc_accuracy_from_output_fn_dict[RO_TYPE]

    all_train_accs = np.zeros(NUM_EPOCHS)
    all_test_accs = np.zeros(NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        train_accs = np.zeros(NUM_BATCHES)
        pbar = tqdm.tqdm(gen_train(), total=NUM_BATCHES)
        for i, np_batch in enumerate(pbar):
            inp_spikes, labels = jax_create_batch(np_batch, NUM_CLASSES, num_neurons=np.prod(IMAGE_DIMS))
            initial_state = init_states_fn(BATCHSIZE)
            params, opt_state, loss, grads, aux = jax.jit(update_fn)(params, thresholds, alphas, initial_state, inp_spikes, labels, opt_state)

            states, out_spikes = aux
            train_accs[i] = jax.jit(calc_accuracy_from_output_fn)(out_spikes[-1], np_batch["targets"])
            pbar.set_description(f"Training Loss {loss:3.6} | Training accuracy {train_accs[0] if i==0 else np.mean(train_accs[max((i-10),0):(i+1)]):2.2%}: | Test accuracy - | ")

            if args.use_thresh_scheduler:
                thresholds = threshold_scheduler(args.thresh_scheduler_mul, thresholds, thresholds_target)

        initial_state = init_states_fn(BATCHSIZE_TEST)
        test_accs = np.zeros(num_samples_test//BATCHSIZE_TEST)
        for i, np_batch in enumerate(gen_test()):
            inp_spikes, labels = jax_create_batch(np_batch, NUM_CLASSES, num_neurons=np.prod(IMAGE_DIMS))
            test_accs[i] = jax.jit(ft.partial(calc_accuracy, calc_accuracy_from_output_fn=calc_accuracy_from_output_fn, sparse_sizes=SPARSE_SIZES))(params, thresholds, alphas, initial_state, inp_spikes, np_batch["targets"])
        pbar.set_description(f"Training Loss {loss} | Training accuracy {np.mean(train_accs):2.2%}: | Test accuracy {np.mean(test_accs):2.2%} | ")

        all_train_accs[epoch] = np.mean(train_accs)
        all_test_accs[epoch] = np.mean(test_accs)

        print("Test accuracy", np.mean(test_accs))
        print("mean_activity", calc_mean_activity(out_spikes))
        if use_wandb:
            wandb.log({'train/loss':loss, "epoch":epoch, "train/acc":np.mean(train_accs), "train/max_acc": np.max(all_train_accs), "test/acc":np.mean(test_accs), "test/max_acc": np.max(all_test_accs)})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SNN training optionally using the IPU and sparse implementations.")
    parser.add_argument('--use_sparse', type=int, default=1, help="Whether to use the IPU (default is `1` therefore `True`).")
    parser.add_argument('--batchsize', type=int, default=256, help="batchsize to use for training, default is 48.")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for optimizer, default `1e-2`.")
    parser.add_argument('--use_lr_scheduler', type=int, default=1, help="Whether to use the learning rate scheduler (default is `1` therefore `True`).")
    parser.add_argument('--second_threshold', type=float, default=0.9, help="Second threshold, default `0.9`.")
    parser.add_argument('--dataset_name', type=str, default="NMNIST", help="dataset name, in ['NMNIST' (default), 'DVSGesture', 'SHD'].")
    parser.add_argument('--max_activity', type=float, default=0.01, help="Max activity in case of 'multi_layer'-mode.")
    parser.add_argument('--num_hidden_layers', type=int, default=3, help="Number of IPUs to use, default `1`.")
    parser.add_argument('--sparse_size_inp', type=int, default=32, help="sparse size for input.")
    # parser.add_argument('--num_neurons_per_tile', type=int, default=2, help="The maximal number of neurons per Tile.")
    parser.add_argument('--seq_len', type=int, default=250, help="The sequence length.")
    parser.add_argument('--num_epochs', type=int, default=25, help="The number of epochs to train the model.")
    parser.add_argument('--batchsize_test', type=int, default=2000, help="The batchsize used for validation.")
    parser.add_argument('--use_wandb', type=int, default=0, help="Whether to use wandb for logging.")
    parser.add_argument('--use_bias', type=int, default=1, help="Whether to use bias.")
    parser.add_argument('--use_lsuv', type=int, default=1, help="Whether to use LSUV.")
    parser.add_argument('--act_rate', type=float, default=0.05, help="Activity rate to use for LSUV init.")
    parser.add_argument('--ro_type', type=str, default="linear_ro", help="Readout type one of 'spike_sum' or 'linear_ro'.")
    parser.add_argument('--ro_int', type=int, default=-1, help="Readout interval to use for prediction and loss calculation.")
    parser.add_argument('--use_thresh_scheduler', type=int, default=1, help="Whether to use a threshold schedule.")
    parser.add_argument('--thresh_scheduler_mul', type=float, default=0.5, help="Multiply factor used in threshold scheduler.")
    parser.add_argument('--use_aug', type=int, default=1, help="Whether to use data agumentation during training (defualt = 0 (False)).")
    parser.add_argument('--use_crop', type=int, default=0, help="Whether to crop data (only used for DVSGesture) (defualt = 0 (False)).")
    parser.add_argument('--root_path_data', type=str, default="/Data/pgi-15/finkbeiner/datasets/", help="Root path to datasets.")
    # parser.add_argument('--num_tests_per_epoch', type=int, default=0.01, help="How often per epoch an eval.")

    args = parser.parse_args()

    main(args)