import sys
import functools as ft
import numpy as np
import jax
# Global flag to set a specific platform, must be used at startup.

from jax._src.lax.slicing import dynamic_slice_p

print(dynamic_slice_p.abstract_eval)

import time

import jax.numpy as jnp
import jax.random as jrandom
from xla_gen_spike_vector import get_gen_spike_vector_fn
from xla_spike_vector_matmul import get_spike_vector_matmul_fn

print(dynamic_slice_p.abstract_eval)
# sys.exit()
# standard_primitive(
#     _dynamic_slice_shape_rule, _dynamic_slice_dtype_rule, 'dynamic_slice',
#     weak_type_rule=_argnum_weak_type(0))

# sys.exit()

gen_spike_vector = get_gen_spike_vector_fn(
    op_name='gen_spike_vector',
    so_file="../lib/gen_spike_vector_from_dense/libgen_sparse_spikes_gpu.so",
    fn_name='gen_spike_vector_gpu_f32',
    platform='gpu',
)


spike_vector_matmul = get_spike_vector_matmul_fn(
    op_name='spike_vector_matmul',
    so_file="../lib/spike_vector_matmul/libspike_vector_matmul_gpu.so",
    fn_name='spike_vector_matmul_gpu_f32',
    platform='gpu',
)


@jax.jit
def fn_sparse(mat, sparse_spike_vector):
    return spike_vector_matmul(mat, sparse_spike_vector)

@jax.jit
def fn_dense(mat, dense_spike_vector):
    return (dense_spike_vector @ mat)

@jax.jit
def fn_sparse_scan(mat, spike_vectors):
    def fn(state, spike_vector):
        return state, spike_vector_matmul(mat, spike_vector)
    return jax.lax.scan(fn, (), spike_vectors)

@jax.jit
def fn_dense_scan(mat, dense_spike_vectors):
    def fn(state, dense_spike_vector):
        return state, dense_spike_vector @ mat
    return jax.lax.scan(fn, (), dense_spike_vectors)


def test_fn(fn, mats, spike_vectors, *, dirname):

    res = fn(mats[0], spike_vectors[0])
    jax.block_until_ready(res)

    start = time.time()
    for i in range(len(mats)):
        res = fn(mats[i], spike_vectors[i])
        jax.block_until_ready(res)
    # for mat, spike_vector in zip(mats, spike_vectors):
    #     # res = fn(mats[1], spike_vectors[1])
    #     res = 
    end = time.time()
    return end - start

def test_fn_scan(fn, mats, *args, **kwargs):
    
    res = fn(mats[0], *args, **kwargs)
    jax.block_until_ready(res)
    start = time.time()
    for mat in mats:
        res = fn(mat, *args, **kwargs)
        jax.block_until_ready(res)
    end = time.time()
    return end - start



num_states = 1024
num_rows = num_states
num_cols = 256
batch_size = 1024
max_num_spikes = 128

num_repeats = 100

gen_spike_vector_fixed = ft.partial(gen_spike_vector, max_num_spikes=max_num_spikes)

@jax.jit
@jax.value_and_grad
def fn_sparse_scan_with_gen(mat, states, thresholds):
    def fn(_, state):
        spike_vector = gen_spike_vector_fixed(state, thresholds)
        return _, spike_vector_matmul(mat, spike_vector)
    return jax.lax.scan(fn, (), states)[1].sum()

@jax.jit
@jax.value_and_grad
def fn_dense_scan_with_gen(mat, states, thresholds):
    def fn(_, state):
        dense_spike_vector = jnp.heaviside(state - thresholds[0], 0.0)
        # spike_vector = gen_spike_vector_fixed(state, thresholds)
        return _, dense_spike_vector @ mat
    return jax.lax.scan(fn, (), states)[1].sum()


states_shape = (num_repeats, batch_size, num_states)

key = jrandom.PRNGKey(42)
keys = jrandom.split(key, 4)
states = jrandom.normal(keys[0], states_shape, dtype=jnp.float32)*0.55
thresholds = jnp.asarray([1.0, 0.85], dtype=jnp.float32)
# matrix = jax.device_put(jrandom.uniform(keys[4], (num_repeats, num_rows, num_cols), dtype=jnp.float32) - 0.5)
matrix = [jax.device_put(jrandom.uniform(keys[4], (num_rows, num_cols), dtype=jnp.float32) - 0.5) for i in range(num_repeats)]

spike_vectors = [jax.device_put(gen_spike_vector_fixed(states[i], thresholds)) for i in range(num_repeats)]
jax.block_until_ready(spike_vectors)
spike_vectors_dense = [jax.device_put(jnp.heaviside(states[i] - thresholds[0], 0.0)) for i in range(num_repeats)]
jax.block_until_ready(spike_vectors_dense)
spike_vectors_dense_single_mat = jnp.heaviside(states - thresholds[0], 0.0)
# spike_vectors_dense = jnp.heaviside(states - thresholds[0], 0.0)
spike_vectors_dense_th1 = jnp.heaviside(states - thresholds[1], 0.0)

def stack_sparse_spike_vectors(spike_vectors):
    comb_spike_data = jnp.stack([spike_vector.comb_spike_data for spike_vector in spike_vectors])
    aval = spike_vectors[0].aval.update(stack_size = len(spike_vectors))
    return type(spike_vectors[0])(comb_spike_data=comb_spike_data, aval=aval)

mean_sparsity = spike_vectors_dense_single_mat.mean()
mean_sparsity_comb = spike_vectors_dense_th1.mean()
mean_sparsity_grad = mean_sparsity_comb - mean_sparsity
print("mean_sparsity:", mean_sparsity)
print("mean_sparsity_grad:", mean_sparsity_grad)
print("mean_sparsity_comb:", mean_sparsity_comb)
print("max allowed sparsity:", max_num_spikes/num_states)
# # sys.exit()

stacked_spike_vectors = stack_sparse_spike_vectors(spike_vectors)
print(stacked_spike_vectors)
print(stacked_spike_vectors[0])
print(stacked_spike_vectors.shape)
print(stacked_spike_vectors[0].shape)
# sys.exit()

# mat = matrix[0]
# fn_sparse(matrix[0], stacked_spike_vectors[0])
# fn_sparse_scan(mat, stacked_spike_vectors)
# fn_sparse_scan_with_gen(mat, states, thresholds)
# fn_dense_scan(matrix[0], spike_vectors_dense_single_mat)


# print("\nSUCESS")
# sys.exit()

# test_sparse = fn_sparse(matrix[0], spike_vectors[0])
# test_dense = fn_dense(matrix[0], spike_vectors_dense[0])

# test_sparse = fn_sparse_scan_with_gen(matrix[0], states, thresholds)
# test_dense = fn_dense_scan_with_gen(matrix[0], states, thresholds)

# sys.exit()
# jax.profiler.start_trace("./tmp/tensorboard/")
# time_dense = test_fn(fn_dense, matrix, spike_vectors_dense, dirname="dense")
# time_sparse = test_fn(fn_sparse, matrix, spike_vectors, dirname="sparse")

time_sparse = test_fn_scan(fn_sparse_scan_with_gen, matrix[:1], states, thresholds)
time_dense = test_fn_scan(fn_dense_scan_with_gen, matrix[:1], states, thresholds)

# jax.profiler.stop_trace()
print()
print("time_sparse =", time_sparse)
print("time_dense  =", time_dense)
print("time_dense/time_sparse =", time_dense/time_sparse)