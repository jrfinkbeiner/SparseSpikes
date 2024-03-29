import sys
import functools as ft
import numpy as np
import jax
# Global flag to set a specific platform, must be used at startup.

import jax.numpy as jnp
import jax.random as jrandom

from sparsespikes.jax_interface import SparseSpikeVector, gen_spike_vector, spike_vector_matmul

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

num_states = 8
num_rows = num_states
num_cols = 8
batch_size = 4
max_num_spikes = 8
seq_len = 100

states_shape = (seq_len, batch_size, num_states)

key = jrandom.PRNGKey(42)
keys = jrandom.split(key, 4)
states = jrandom.uniform(keys[0], states_shape, dtype=jnp.float32)*1.5
thresholds = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
matrix = jrandom.uniform(keys[1], (num_rows, num_cols), dtype=jnp.float32) - 0.5
# matrix = jnp.ones((num_rows, num_cols), dtype=jnp.float32)

gen_spike_vector_fixed = ft.partial(gen_spike_vector, max_num_spikes=max_num_spikes)

def test_fn_dense(weights, states_pre, threshold_pre):
    def fn(carry, state_pre):
        return carry, heaviside_with_super_spike_surrogate(state_pre, threshold_pre) @ weights
    carry_final, out = jax.lax.scan(fn, (), states_pre)
    return out 

def test_fn_sparse(weights, states_pre, threshold_pre):
    def fn(carry, state_pre):
        spike_vector = gen_spike_vector_fixed(state_pre, threshold_pre, max_num_spikes=max_num_spikes)
        return carry, spike_vector_matmul(weights, spike_vector)
    carry_final, out = jax.lax.scan(fn, (), states_pre)
    return out
    
res_sparse = test_fn_sparse(matrix, states, thresholds) 
res_dense = test_fn_dense(matrix, states, thresholds) 
print("Correct batched: ", np.allclose(res_sparse, res_dense))

res_sparse = jax.vmap(test_fn_sparse, in_axes=(None, 1, None))(matrix, states, thresholds) 
res_dense = jax.vmap(test_fn_dense, in_axes=(None, 1, None))(matrix, states, thresholds) 
print("Correct vmap: ", np.allclose(res_sparse, res_dense))

grads_sparse = jax.grad(lambda x,y,z: test_fn_sparse(x,y,z).sum(), argnums=(0, 1))(matrix, states, thresholds)
grads_dense = jax.grad(lambda x,y,z: test_fn_dense(x,y,z).sum(), argnums=(0, 1))(matrix, states, thresholds)
correct_grads = jax.tree_util.tree_map(lambda x, y: np.allclose(x, y), grads_sparse, grads_dense)
print("Correct grad: ", correct_grads)