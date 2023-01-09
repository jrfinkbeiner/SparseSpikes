import sys
from typing import Union, NamedTuple, Optional, List
import functools as ft
import numpy as np
import jax
# Global flag to set a specific platform, must be used at startup.

import jax.numpy as jnp
import jax.random as jrandom
# from xla_gen_spike_vector import get_gen_spike_vector_fn
# from xla_spike_vector_matmul import get_spike_vector_matmul_fn
# from xla_spike_vector import SparseSpikeVector

# gen_spike_vector = get_gen_spike_vector_fn(
#     op_name='gen_spike_vector',
#     so_file="../../lib/gen_spike_vector_from_dense/libgen_sparse_spikes_gpu.so",
#     fn_name='gen_spike_vector_gpu_f32',
#     platform='gpu',
# )

# spike_vector_matmul = get_spike_vector_matmul_fn(
#     op_name='spike_vector_matmul',
#     so_file="../../lib/spike_vector_matmul/libspike_vector_matmul_gpu.so",
#     fn_name='spike_vector_matmul_gpu_f32',
#     platform='gpu',
# )

from sparsespikes.jax_interface import SparseSpikeVector, check_is_sparse_spikes_type, gen_spike_vector, spike_vector_matmul

def get_heaviside_with_super_spike_surrogate(beta=10.):
    @jax.custom_jvp
    def heaviside_with_super_spike_surrogate(state, thresholds):
        return jnp.heaviside(state-thresholds[0], 0)

    @heaviside_with_super_spike_surrogate.defjvp
    def f_jvp(primals, tangents):
        state, thresholds = primals
        state_dot, thresh_dot = tangents
        primal_out = heaviside_with_super_spike_surrogate(state, thresholds)
        tangent_out = 1.0 / (beta*jnp.abs(state-thresholds[0]) + 1.0)**2 * state_dot * jnp.heaviside(state-thresholds[1], 0)
        return primal_out, tangent_out

    return heaviside_with_super_spike_surrogate

heaviside_with_super_spike_surrogate = get_heaviside_with_super_spike_surrogate()

def get_gen_spike_fn(use_sparse: bool):
    def gen_spike_dense(state, thresholds, *, max_num_spikes: Optional[int] = None):
        """Function to generate dense spikes from a given state."""
        return heaviside_with_super_spike_surrogate(state, thresholds)
    def gen_spike_sparse(state, thresholds, *, max_num_spikes: Optional[int] = None):
        """Function to generate sparse spikes from a given state."""
        if max_num_spikes is None:
            max_num_spikes = state.shape[-1]
        return gen_spike_vector(state, thresholds, max_num_spikes=max_num_spikes)
    spike_fn = gen_spike_sparse if use_sparse else gen_spike_dense
    return spike_fn 

class LIFDenseNeuronState(NamedTuple):
    """
    Generic Module for storing the state of an RNN/SNN. 
    Each state variable is a union of a numpy array and a 
    jax numpy array to make backpropagation possible.
    """
    # TODO change docstring
    U: Union[np.ndarray, jnp.ndarray]
    I: Union[np.ndarray, jnp.ndarray]

def linear_layer(weights, bias, inp):
    """Simple implementation of a fully connected layer."""
    if isinstance(inp, jnp.ndarray):
        ret_val = inp @ weights
    else:
        ret_val = spike_vector_matmul(weights, inp)
    if bias is not None:
        ret_val += bias 
    return ret_val

def linear_layer_dense(weights, bias, inp):
    """Simple implementation of a fully connected layer."""
    ret_val = inp @ weights
    if bias is not None:
        ret_val += bias 
    return ret_val

def linear_layer_sparse(weights, bias, inp):
    """Simple implementation of a fully connected layer."""
    ret_val = spike_vector_matmul(weights, inp)
    if bias is not None:
        ret_val += bias 
    return ret_val

def reset_with_dense_spikes(state, out_spikes):
    return state * jax.lax.stop_gradient(1-out_spikes)

def reset_from_state(state, thresholds):
    return state * jax.lax.stop_gradient(1-jnp.heaviside(state-thresholds[0], 0))

def get_reset_fn(use_sparse: bool):
    def reset_fn(state, thresholds, out_spikes):
        if use_sparse:
            # TODO use sparse reset
            return reset_from_state(state, thresholds)
        else:
            return reset_with_dense_spikes(state, out_spikes)
    return reset_fn

def get_lif_step(use_sparse: bool):
    """Function to return the correct LIF step function."""
    liner_layer_fn = linear_layer_sparse if use_sparse else linear_layer_dense
    gen_spike_fn = get_gen_spike_fn(use_sparse)
    reset_fn = get_reset_fn(use_sparse)

    def lif_step(weights, alpha, beta, state, Sin_t, thresholds, *, use_output_spikes: bool = True, max_num_spikes: Optional[int] = None):
        """Simple implementation of a layer of leaky integrate-and-fire neurons."""
        U, I = state
        fc_weight, fc_bias = weights
        U = alpha*U + (1-alpha)*(20.0*I) # I is weighted by a factor of 20
        I = beta*I + (1-beta) * liner_layer_fn(fc_weight, fc_bias, Sin_t)
        if use_output_spikes:
            out_val = gen_spike_fn(U, thresholds, max_num_spikes=max_num_spikes)
            U = reset_fn(U, thresholds, out_val)
        else:
            out_val = U
        state_new = LIFDenseNeuronState(U, I)
        return state_new, out_val
    return lif_step

def init_weights(rng: np.random.Generator, dim_in: int, dim_out: int, use_bias: bool):
    """A simple function to initialize the weights of a fully connected layer."""
    lim = (6/(dim_in+dim_out))**0.5
    weights = rng.uniform(-lim, lim, (dim_in, dim_out))
    bias = np.zeros(dim_out) if use_bias else None
    return weights, bias

def init_state(shape):
    """Function to initialize the state variables of our LIF layer."""
    return LIFDenseNeuronState(*[np.zeros(shape) for _ in range(2)])

def lif_network(weights, thresholds, alphas, betas, initial_state, inp_spikes):
    """
    Function to initialize a stack of LIF layers from given weights matrices etc.
    It also computed the forward pass of the network for given input spikes.
    """
    use_sparse = check_is_sparse_spikes_type(inp_spikes)
    lif_step_fn = get_lif_step(use_sparse)
    print("\nuse_sparse", use_sparse)
    print()
    def step_fn_lif_network(states, spikes):
        """Performes a forward pass for the entire LIF network."""
        all_states, all_spikes = [], []
        for ilay,params in enumerate(zip(weights, alphas, betas, states)):
            print("\nilay", ilay)
            use_output_spikes = ilay < len(weights)-1
            new_state, spikes = lif_step_fn(*params, spikes, thresholds, use_output_spikes=use_output_spikes)
            all_states.append(new_state)
            all_spikes.append(spikes)
        return all_states, all_spikes
    final_state, out_spikes = jax.lax.scan(step_fn_lif_network, initial_state, inp_spikes)
    return final_state, out_spikes

def init_network_weights(rng, dims, use_bias_fc):
    """Function to initialize the weights of the entire network."""
    num_layers = len(dims)-1
    all_weights = []
    for ilay in range(num_layers):
        fc_weights = init_weights(rng, dims[ilay], dims[ilay+1], use_bias_fc)
        all_weights.append(fc_weights)
    return all_weights

def init_network_states(batchsize, state_dims):
    """Function to initilize the states of every layer."""
    return [init_state((batchsize, dim)) for dim in state_dims]

def create_one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def one_hot_crossentropy(target_one_hot, pred):
    """
    Function to calculate the softmax cross-entropy of a batch of 
    one-hot encoded target and the network output.
    """
    return -jnp.sum(target_one_hot*jax.nn.log_softmax(pred)) / len(target_one_hot)

def sum_and_crossentropy(one_hot_target, y_pred):
    """Sum the spikes over the sequence length and then calculate crossentropy."""
    sum_spikes = y_pred.sum(axis=0) # y_pred shape: (seq_len, batch, neurons)
    return one_hot_crossentropy(one_hot_target, sum_spikes)

loss_func = sum_and_crossentropy

def calc_loss_single(weights, thresholds, alphas, betas, initial_state, inp_spikes, labels):
    """Function that calculates the loss for a single sample."""
    final_state, out_spikes = lif_network(weights, thresholds, alphas, betas, initial_state, inp_spikes)
    final_layer_out_spikes = out_spikes[-1]
    return loss_func(labels, final_layer_out_spikes)

def calc_loss_batch(weights, thresholds, alphas, betas, initial_state, inp_spikes, labels):
    """
    Function that calculates the loss for a batch of samples.
    For this, we use vectorization through jax.vmap(...) which
    accelerates the computations.
    """
    loss_vals = jax.vmap(calc_loss_single, in_axes=(None, None, None, None, 0, 1, 0))(
        weights, thresholds, alphas, betas, initial_state, inp_spikes, labels)
    return loss_vals.sum()


def main():
    Nc = 2 # Number of classes
    N = [4, 8, Nc] # List of number of neurons per layer
    T = 5 # Number of timesteps per epoch
    BATCHSIZE = 12
    SEED = 42 

    rng = np.random.default_rng(SEED)
    keys = jrandom.split(jrandom.PRNGKey(rng.integers(999999)), 2)
    weights = init_network_weights(rng, N, False)
    NUM_LAYERS = len(N)-1
    alphas, betas = [0.95]*NUM_LAYERS, [0.9]*NUM_LAYERS

    initial_state = init_network_states(BATCHSIZE, N[1:])
    target = rng.integers(0, Nc, size=(BATCHSIZE,))
    targets_one_hot = create_one_hot(target, Nc, dtype=jnp.float32)

    inp_states = jrandom.uniform(keys[0], (T, BATCHSIZE, N[0]), dtype=jnp.float32)*1.5
    thresholds = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
    inp_spikes_dense = jnp.asarray(inp_states > thresholds[0], dtype=jnp.float32)
    inp_spikes_sparse_list = [gen_spike_vector(inp_stat, thresholds, max_num_spikes=N[0]) for inp_stat in inp_states] 

    def stack_sparse_spike_vectors(sparse_spike_vectors: List[SparseSpikeVector]):
        """Stacks the sparse spike vectors into a SparseSpikeVector instance."""
        comb_spike_data = jnp.stack([spike_vec.comb_spike_data for spike_vec in sparse_spike_vectors], axis=0)
        stacked_aval = sparse_spike_vectors[0].aval.update(stack_size=len(sparse_spike_vectors))
        return SparseSpikeVector(comb_spike_data=comb_spike_data, aval=stacked_aval)

    # print(inp_spikes_sparse.shape)
    print(f"inp_spikes_dense.shape: {inp_spikes_dense.shape}")
    print(f"inp_spikes_sparse.shape: {inp_spikes_sparse_list[0].shape}")
    inp_spikes_sparse = stack_sparse_spike_vectors(inp_spikes_sparse_list)
    # inp_spikes_sparse = jax.vmap(ft.partial(gen_spike_vector, max_num_spikes=N[0]), in_axes=(0,None))(inp_states, thresholds)
    print(f"inp_spikes_sparse.shape: {inp_spikes_sparse.shape}")



    # jaxpr_dense = jax.make_jaxpr(calc_loss_batch)(weights, thresholds, alphas, betas, initial_state, inp_spikes_dense, targets_one_hot)
    # with open("jaxpr_dense.txt", "w") as f:
    #     f.write(str(jaxpr_dense))

    # jaxpr_sparse = jax.make_jaxpr(calc_loss_batch)(weights, thresholds, alphas, betas, initial_state, inp_spikes_sparse, targets_one_hot)
    # with open("jaxpr_sparse.txt", "w") as f:
    #     f.write(str(jaxpr_sparse))
    # sys.exit()


    print("\nCalculating loss and gradients...")
    print("\nDense:")
    loss_dense, grads_dense = jax.jit(jax.value_and_grad(calc_loss_batch))(weights, thresholds, alphas, betas, initial_state, inp_spikes_dense, targets_one_hot)
    print("\nSparse:")
    loss_sparse, grads_sparse = jax.jit(jax.value_and_grad(calc_loss_batch))(weights, thresholds, alphas, betas, initial_state, inp_spikes_sparse, targets_one_hot)
    # print("\nSparse 2:")
    # loss_sparse, grads_sparse = jax.value_and_grad(calc_loss_batch)(weights, thresholds, alphas, betas, initial_state, inp_spikes_sparse, targets_one_hot)

    # sys.exit()
    print(f"Loss dense: {loss_dense:.3f}")
    print(f"Loss sparse: {loss_sparse:.3f}")
    print("\nGradients:")
    import pprint
    pprint.pprint(grads_dense)
    pprint.pprint(grads_sparse)
    # correct_grads = jax.tree_util.tree_map(lambda x,y: , grads_dense, [grads_sparse])
    correct_grads = jax.tree_util.tree_map(lambda x, y: np.allclose(x, y), grads_sparse, grads_dense)
    print("\nCorrect gradients:", correct_grads)

if __name__ == "__main__":
    main()