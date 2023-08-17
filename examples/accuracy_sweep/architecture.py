import sys
from typing import Union, NamedTuple, Optional, List
import functools as ft
import numpy as np
import jax
# Global flag to set a specific platform, must be used at startup.

import jax.numpy as jnp
import jax.random as jrandom

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

    def lif_step(weights, alpha, state, Sin_t, thresholds, *, use_output_spikes: bool = True, max_num_spikes: Optional[int] = None):
        """Simple implementation of a layer of leaky integrate-and-fire neurons."""
        fc_weight, fc_bias = weights
        
        mul_fac = 20
        beta = 0.9
        # U = alpha*state.U + (1-alpha)*(mul_fac*state.I) # I is weighted by a factor of 20
        # I = beta*state.I + (1-beta) * liner_layer_fn(fc_weight, fc_bias, Sin_t)
        
        U = alpha*state.U + (1-alpha)*(mul_fac*liner_layer_fn(fc_weight, fc_bias, Sin_t))

        if use_output_spikes:
            out_val = gen_spike_fn(U, thresholds, max_num_spikes=max_num_spikes)
            U = reset_fn(U, thresholds, out_val)
        else:
            out_val = U
        # state_new = LIFDenseNeuronState(U, I)
        state_new = LIFDenseNeuronState(U, state.I)
        return state_new, out_val
    return lif_step

# def custom_init(in_feat, out_feat, dtype):
#     limit = (6/(in_feat))**0.5*(out_feat)**0.2 * self.weight_mul
#     return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype) + 0.001


def init_weights(key, dim_in: int, dim_out: int, use_bias: bool, dtype: np.dtype):
    """A simple function to initialize the weights of a fully connected layer."""
    # lim = (6/(dim_in+dim_out))**0.5*2
    # lim = 0.1 # NMNIST
    lim = 0.01 # SHD
    # lim = (6/(dim_in))**0.5*(dim_out)**0.2
    weights = jax.random.uniform(key, (dim_in, dim_out), dtype=dtype, minval=-lim, maxval=lim)
    bias = jnp.zeros(dim_out, dtype=dtype) if use_bias else None
    return weights, bias

def init_state(shape):
    """Function to initialize the state variables of our LIF layer."""
    return LIFDenseNeuronState(*[np.zeros(shape) for _ in range(2)])
    # return LIFDenseNeuronState(jnp.zeros(shape))

def lif_network(weights, thresholds, alphas, initial_state, inp_spikes, *, sparse_sizes=None):
    """
    Function to initialize a stack of LIF layers from given weights matrices etc.
    It also computed the forward pass of the network for given input spikes.
    """
    linear_readout = True if len(weights) > len(initial_state) else False
    use_sparse = check_is_sparse_spikes_type(inp_spikes)
    lif_step_fn = get_lif_step(use_sparse)
    # workaround until .to_dense() for SparseSpikes is implemented
    liner_layer_fn = linear_layer_sparse if use_sparse else linear_layer_dense
    # print("\nuse_sparse", use_sparse)
    def step_fn_lif_network(states, spikes):
        """Performes a forward pass for the entire LIF network."""
        all_states, all_spikes = [], []
        snn_weights = weights[:-1] if linear_readout else weights
        for ilay,params in enumerate(zip(snn_weights, alphas, states)):
            use_output_spikes = not (linear_readout and (ilay >= (len(snn_weights)-1)))
            max_num_spikes = sparse_sizes[ilay] if use_output_spikes and use_sparse else None
            # print(ilay, use_output_spikes, max_num_spikes)
            new_state, spikes = lif_step_fn(*params, spikes, thresholds, use_output_spikes=use_output_spikes, max_num_spikes=max_num_spikes)
            all_states.append(new_state)
            all_spikes.append(spikes)

        if not linear_readout:
            last_state = states[-1]
            last_spikes_dense = liner_layer_fn(jnp.eye(last_state.U.shape[-1]), None, spikes)
            last_state = LIFDenseNeuronState(last_state.U + last_spikes_dense, jnp.zeros_like(last_state.U))
            all_states.append(last_state)
            all_spikes.append(last_spikes_dense)
        return all_states, (all_states, all_spikes)
    final_state, (states, out_spikes) = jax.lax.scan(step_fn_lif_network, initial_state, inp_spikes)
    if linear_readout:
        out_spikes.append(jax.vmap(linear_layer_dense, in_axes=(None, None, 0))(weights[-1][0], None, states[-1].U))
    return states, out_spikes

def init_network_weights(key, dims, use_bias_fc, dtype=np.float32):
    """Function to initialize the weights of the entire network."""
    num_layers = len(dims)-1
    keys = jax.random.split(key, num_layers)
    all_weights = []
    for ilay,keyi in enumerate(keys):
        fc_weights = init_weights(keyi, dims[ilay], dims[ilay+1], use_bias_fc, dtype)
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

def sum_and_crossentropy(one_hot_target, y_pred, *, sum_first=True):
    """Sum the spikes over the sequence length and then calculate crossentropy."""
    # sum_spikes = y_pred.sum(axis=0) # y_pred shape: (seq_len, batch, neurons)
    if sum_first:
        y_pred = y_pred.sum(axis=0)
        res = one_hot_crossentropy(one_hot_target, y_pred)
    else:
        res = jax.vmap(one_hot_crossentropy, in_axes=(None, 0))(one_hot_target, y_pred)
        res = res.sum(axis=0)
    return res

loss_func = sum_and_crossentropy

def calc_loss_single(weights, thresholds, alphas, initial_state, inp_spikes, labels, loss_func, *, sparse_sizes=None):
    """Function that calculates the loss for a single sample."""
    final_state, out_spikes = lif_network(weights, thresholds, alphas, initial_state, inp_spikes, sparse_sizes=sparse_sizes)
    final_layer_out_spikes = out_spikes[-1]
    return loss_func(labels, final_layer_out_spikes), final_state, out_spikes

def calc_loss_batch(weights, thresholds, alphas, initial_state, inp_spikes, labels, loss_func=loss_func, sparse_sizes=None):
    """
    Function that calculates the loss for a batch of samples.
    For this, we use vectorization through jax.vmap(...) which
    accelerates the computations.
    """
    loss_vals, final_state, out_spikes = jax.vmap(ft.partial(calc_loss_single, sparse_sizes=sparse_sizes), in_axes=(None, None, None, 0, 1, 0, None), out_axes=(0,0,1))(
        weights, thresholds, alphas, initial_state, inp_spikes, labels, loss_func)
    return loss_vals.sum(), (final_state, out_spikes)