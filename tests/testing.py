import sys
import functools as ft
import numpy as np
import jax
# Global flag to set a specific platform, must be used at startup.

import jax.numpy as jnp
import jax.random as jrandom
from xla_gen_spike_vector import get_gen_spike_vector_fn
from xla_spike_vector_matmul import get_spike_vector_matmul_fn
from xla_spike_vector import SparseSpikeVector

# jax.config.update('jax_platform_name', 'cpu')
# gen_spike_vector = get_gen_spike_vector_fn(
#     op_name='gen_spike_vector',
#     so_file="../lib/gen_spike_vector_from_dense/libgen_sparse_spikes.so",
#     fn_name='gen_spike_vector_cpu_f32',
#     platform='cpu',
# )

gen_spike_vector = get_gen_spike_vector_fn(
    op_name='gen_spike_vector',
    so_file="../lib_withGrad/gen_spike_vector_from_dense/libgen_sparse_spikes_gpu.so",
    fn_name='gen_spike_vector_gpu_f32',
    platform='gpu',
)

spike_vector_matmul = get_spike_vector_matmul_fn(
    op_name='spike_vector_matmul',
    so_file="../lib_withGrad/spike_vector_matmul/libspike_vector_matmul_gpu.so",
    fn_name='spike_vector_matmul_gpu_f32',
    platform='gpu',
)

def spike_vector_matmul_np_unbatched(matrix, spike_ids, num_spikes, res):
    """Numpy implementation of spike_vector_matmul."""
    for j in range(num_spikes):
        res[:] += matrix[spike_ids[j]]

def spike_vector_matmul_np(matrix, spike_vector):
    """Numpy implementation of spike_vector_matmul."""
    spike_ids = np.array(spike_vector.spike_ids)
    num_spikes = np.array(spike_vector.num_spikes)
    matrix = np.array(matrix)

    if spike_vector.batched:
        res = np.zeros((spike_vector.batchsize, matrix.shape[1]), dtype=matrix.dtype)
        for ibatch in range(spike_vector.batchsize):
            spike_vector_matmul_np_unbatched(matrix, spike_ids[ibatch], num_spikes[0, ibatch], res[ibatch])
    else:
        res = np.zeros((matrix.shape[1]), dtype=matrix.dtype)
        spike_vector_matmul_np_unbatched(matrix, spike_ids, num_spikes[0], res)
    return res


num_states = 8
num_rows = num_states
num_cols = 8
batch_size = None
batched = True if batch_size is not None else False
max_num_spikes = 8

states_shape = (batch_size, num_states) if batched else (num_states,)

key = jrandom.PRNGKey(42)
keys = jrandom.split(key, 4)
states = jrandom.uniform(keys[0], states_shape, dtype=jnp.float32)*1.5
thresholds = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
matrix = jrandom.uniform(keys[4], (num_rows, num_cols), dtype=jnp.float32) - 0.5
# matrix = jnp.ones((num_rows, num_cols), dtype=jnp.float32)

print(matrix)

# spike_ids = jrandom.choice(keys[1], num_states, (max_num_spikes, ), replace=False).astype(jnp.uint32)
# num_spikes = jrandom.uniform(keys[2], (2,), maxval=max_num_spikes+1).astype(jnp.uint32)
# spike_vector_1 = SparseSpikeVector(spike_ids, num_spikes)
# print(spike_vector_1)
# sys.exit()
spike_vector = gen_spike_vector(states, thresholds, max_num_spikes=max_num_spikes)

print(spike_vector)

print(spike_vector.spike_grads)
# sys.exit()
gen_spike_vector_fixed = ft.partial(gen_spike_vector, max_num_spikes=max_num_spikes)
# spike_vector = jax.vmap(gen_spike_vector_fixed, in_axes=(0, None), out_axes=0)(states, thresholds)
dense_spike_vector = jnp.heaviside(states - thresholds[0], 0.0)

# print()
# print(spike_vector)

# print(spike_vector.aval)
# print(spike_vector.dtype)
# spike_vector.aval = spike_vector.aval.update(dtype=jnp.float32)
# print(spike_vector.aval)
# print(spike_vector.dtype)
# # sys.exit()


# print(spike_vector)
# # sys.exit()

# print("\n start spike_vector_matmul")
# result = spike_vector_matmul(matrix, spike_vector)
print("\n start matrix grad spike_vector_matmul")
result, grad = jax.value_and_grad(lambda mat,sp: spike_vector_matmul(mat, sp).sum(), argnums=(0, 1))(matrix, spike_vector)
result_dense, grad_dense = jax.value_and_grad(lambda mat,sp: (sp @ mat).sum(), argnums=(0, 1))(matrix, dense_spike_vector)
# jac = jax.jacrev(spike_vector_matmul, argnums=(0, 1))(matrix, spike_vector)

print("\n start spike_vector_matmul_np")
res_np = spike_vector_matmul_np(matrix, spike_vector).sum()

print("\ngrad sparse")
print("result", result)
print("matrix_grad = ",grad[0])
print("spike_vector_grad = ",grad[1])
print("spike_vector = " , spike_vector)

print("\ngrad dense")
print("result", result_dense)
print("matrix_grad = ",grad_dense[0])
print("spike_vector_grad = ",grad_dense[1])
print("dense_spike_vector = " , dense_spike_vector)


# print("\nnp")
# print(spike_vector.aval)
# print(spike_vector.num_spikes.shape)
# # sys.exit()
# print()

# print("spike_vector = " , spike_vector)
# print("spike_vector_grad = ", grad[1])
# if not batched:
#     spike_ids_np = np.array(spike_vector.spike_ids[:int(spike_vector.num_spikes[1])])
#     print("spike_vector_grad = ", grad_dense[1][spike_ids_np])
# print("dense_spike_vector = " , dense_spike_vector)
# # sys.exit()

# # print(len(jac))
# # print(type(jac))
# # print(result)
# # print(res_np)
# # sys.exit()

# # matrix_grad = grad[0]
# # spike_vector_grad = grad[1]

# # # print(result)
# # # print()
# # # print(spike_vector)
# # print()
# # print(matrix_grad)
# # print(matrix_grad.shape)
# # print()
# # print(spike_vector_grad)
# # print(spike_vector_grad.shape)
# # # sys.exit()

# print("\n----------------------------- gen_spike_vector_fixed -----------------------------")

if batched:
    primals, f_vjp = jax.vjp(jax.vmap(gen_spike_vector_fixed, in_axes=(0, None), out_axes=0), states, thresholds)
else:
    primals, f_vjp = jax.vjp(gen_spike_vector_fixed, states, thresholds)
print("\nprimals")
print(primals)


xbar, ybar = f_vjp(primals)

print("\ngrads")
print(xbar)
print(ybar)
# sys.exit()

# if batched:
#     jac = jax.jacrev(jax.vmap(gen_spike_vector_fixed, in_axes=(0, None), out_axes=0), argnums=0)(states, thresholds)
# else:
# print(states.shape)
# print(thresholds.shape)
# sys.exit()
# jac = jax.jacrev(gen_spike_vector_fixed, argnums=0)(states, thresholds)
# jac = jax.jacrev(gen_spike_vector_fixed, argnums=0)(states, thresholds)
# print(jac)
# print(spike_vector)
# sys.exit()    
    
spike_ids = np.asarray(spike_vector.spike_ids)
num_spikes = np.asarray(spike_vector.num_spikes)

# print(num_spikes)
# print(num_spikes[0])
# print(num_spikes[1])
# sys.exit()

spike_ids_fwd_comp = np.where(states > thresholds[0])
spike_ids_grad_comp = np.where((states > thresholds[1]) * (states <= thresholds[0]))

if batched:
    spike_ids_fwd = [np.empty(np.sum(num_spikes[0]), dtype=np.uint32), []]
    spike_ids_grad = [np.empty(np.sum(num_spikes[1]-num_spikes[0]), dtype=np.uint32), []]
    num_spikes_fwd_start = 0
    num_spikes_fwd_end = 0
    num_spikes_grad_start = 0
    num_spikes_grad_end = 0
    for ibatch in range(batch_size):
        num_spikes_fwd_end += int(num_spikes[0, ibatch])
        num_spikes_grad_end += int(num_spikes[1, ibatch] - num_spikes[0, ibatch])
        spike_ids_fwd[0][num_spikes_fwd_start:num_spikes_fwd_end] = ibatch
        spike_ids_fwd[1].append(np.sort(spike_ids[ibatch, :num_spikes[0, ibatch]]))
        spike_ids_grad[0][num_spikes_grad_start:num_spikes_grad_end] = ibatch
        spike_ids_grad[1].append(np.sort(spike_ids[ibatch, num_spikes[0, ibatch]:num_spikes[1,ibatch]]))
        num_spikes_fwd_start = num_spikes_fwd_end
        num_spikes_grad_start = num_spikes_grad_end

    spike_ids_fwd[1] = np.concatenate(spike_ids_fwd[1])
    spike_ids_grad[1] = np.concatenate(spike_ids_grad[1])
else:
    spike_ids_fwd_comp = spike_ids_fwd_comp[0]
    spike_ids_grad_comp = spike_ids_grad_comp[0]
    spike_ids_fwd = np.sort(spike_ids[:num_spikes[0]])
    spike_ids_grad = np.sort(spike_ids[num_spikes[0]:num_spikes[1]])

print("\nDONE")
print(states)
print(spike_vector)
print(spike_ids)
print(num_spikes)
print("\n")
print("spike_ids_fwd")
print(spike_ids_fwd)
print("spike_ids_fwd_comp")
print(spike_ids_fwd_comp)
print("spike_ids_grad")
print(spike_ids_grad)
print("spike_ids_grad_comp")
print(spike_ids_grad_comp)


correct_fwd = np.array_equal(np.asarray(spike_ids_fwd), np.asarray(spike_ids_fwd_comp))
correct_grad = np.array_equal(np.asarray(spike_ids_grad), np.asarray(spike_ids_grad_comp))

print()
print("Correct forward: ", correct_fwd)
print("Correct gradient:", correct_grad)

print("Correct matmul: ", np.allclose(result, res_np))