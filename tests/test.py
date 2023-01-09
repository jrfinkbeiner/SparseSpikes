import sys
import functools as ft
import numpy as np
import jax
# Global flag to set a specific platform, must be used at startup.

import jax.numpy as jnp
import jax.random as jrandom
# from xla_gen_spike_vector import get_gen_spike_vector_fn
# from xla_spike_vector_matmul import get_spike_vector_matmul_fn
# # from xla_spike_vector import SparseSpikeVector

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
batched = True if batch_size is not None else False
max_num_spikes = 8

states_shape = (batch_size, num_states) if batched else (num_states,)

key = jrandom.PRNGKey(42)
keys = jrandom.split(key, 4)
states = jrandom.uniform(keys[0], states_shape, dtype=jnp.float32)*1.5
thresholds = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
matrix = jrandom.uniform(keys[1], (num_rows, num_cols), dtype=jnp.float32) - 0.5
# matrix = jnp.ones((num_rows, num_cols), dtype=jnp.float32)

gen_spike_vector_fixed = ft.partial(gen_spike_vector, max_num_spikes=max_num_spikes)

def test_fn_dense(weights, state_pre, threshold_pre):
    return heaviside_with_super_spike_surrogate(state_pre, threshold_pre) @ weights

def test_fn_sparse(weights, state_pre, threshold_pre):
    spike_vector = gen_spike_vector_fixed(state_pre, threshold_pre, max_num_spikes=max_num_spikes)
    return spike_vector_matmul(weights, spike_vector)

def matmul_grad_fn_sparse(weights, state_pre, threshold_pre):
    spike_vector = gen_spike_vector_fixed(state_pre, threshold_pre, max_num_spikes=max_num_spikes)
    return spike_vector, jax.grad(lambda x, y: spike_vector_matmul(x, y).sum(), argnums=(0, 1))(weights, spike_vector)

def matmul_grad_fn_dense(weights, state_pre, threshold_pre):
    spike_vector = heaviside_with_super_spike_surrogate(state_pre, threshold_pre)
    return jax.grad(lambda x, y: (y @ x).sum(), argnums=(0, 1))(weights, spike_vector)


res_sparse = test_fn_sparse(matrix, states, thresholds) 
res_dense = test_fn_dense(matrix, states, thresholds) 
print("Correct batched: ", np.allclose(res_sparse, res_dense))

res_sparse = jax.vmap(test_fn_sparse, in_axes=(None, 0, None))(matrix, states, thresholds) 
res_dense = jax.vmap(test_fn_dense, in_axes=(None, 0, None))(matrix, states, thresholds) 
print("Correct vmap: ", np.allclose(res_sparse, res_dense))

# spike_vector, grads_sparse = matmul_grad_fn_sparse(matrix, states, thresholds)
# # grads_dense = matmul_grad_fn_dense(matrix, states, thresholds)
# print(grads_sparse)
# print()
# print(spike_vector)
# # correct_grads = jax.tree_util.tree_map(lambda x, y: np.allclose(x, y), grads_sparse, grads_dense)
# # print("Correct grad matmul: ", correct_grads)

grads_sparse = jax.grad(lambda x,y,z: test_fn_sparse(x,y,z).sum(), argnums=(0, 1))(matrix, states, thresholds)
grads_dense = jax.grad(lambda x,y,z: test_fn_dense(x,y,z).sum(), argnums=(0, 1))(matrix, states, thresholds)
correct_grads = jax.tree_util.tree_map(lambda x, y: np.allclose(x, y), grads_sparse, grads_dense)
print("Correct grad: ", correct_grads)


# print("\ngrads_sparse", grads_sparse)
# print("\ngrads_dense", grads_dense)

# jac_sparse = jax.jacrev(test_fn_sparse, argnums=(0,))(matrix, states, thresholds)
# jac_dense = jax.jacrev(test_fn_dense, argnums=(0,))(matrix, states, thresholds)
# jax.tree_util.tree_multimap(lambda x, y: print("Correct jac: ", np.allclose(x, y)), jac_sparse, jac_dense)


# print("\nres_sparse", res_sparse)
# print("\nres_dense", res_dense)