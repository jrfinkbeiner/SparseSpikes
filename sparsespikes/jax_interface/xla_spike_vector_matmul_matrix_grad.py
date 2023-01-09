import sys
import ctypes
import struct

import functools as ft
import numpy as np
import jax
import jax.numpy as jnp

from jax import core
from jax._src import abstract_arrays
from jax.interpreters import xla
from jax._src.lib import xla_client
from jax.interpreters import ad
from jax.interpreters import batching

from .interface_utils import create_pycapsule

def get_spike_vector_matmul_matrix_grad_fn(op_name: str, so_file: str, fn_name: str, platform: str):

    def spike_vector_matmul_matrix_grad(spike_vector, result_grad):
        """The JAX-traceable way to use the JAX primitive.
        
        Note that the traced arguments must be passed as positional arguments
        to `bind`. 
        """
        return _spike_vector_matmul_matrix_grad_p.bind(spike_vector, result_grad, num_rows=spike_vector.num_neurons)

    def _spike_vector_matmul_matrix_grad_abstract_eval(spike_vector, result_grad, *, num_rows):
        
        if spike_vector.batched:
            assert result_grad.ndim == 2
        else:
            assert result_grad.ndim == 1
        num_cols = result_grad.shape[-1] 
        assert result_grad.dtype == np.dtype("float32")

        out_shape = (num_rows, num_cols)
        return abstract_arrays.ShapedArray(out_shape, result_grad.dtype)

    def _spike_vector_matmul_matrix_grad_xla_translation(ctx, avals_in, avals_out, spike_vector, results_grad, *, num_rows):
 
        spike_vector_shape = avals_in[0]
        result_shape = avals_in[1]
        matrix_grad = avals_out[0]

        batchsize = 1 if result_shape.ndim == 1 else result_shape.shape[0]
        num_rows, num_cols = matrix_grad.shape
        max_num_spikes = spike_vector_shape.max_num_spikes

        # operand_shapes = [xla.aval_to_xla_shape(av) for av in avals_in]
        operands_base = (spike_vector, results_grad)
        operand_shapes = [
            xla_client.Shape.array_shape(arr.dtype, arr.shape, tuple(range(len(arr.shape) - 1, -1, -1))) 
            for arr in [spike_vector_shape, result_shape]
        ]
        matrix_grad_shape = xla_client.Shape.array_shape(matrix_grad.dtype, matrix_grad.shape, tuple(range(len(matrix_grad.shape) - 1, -1, -1)))

        if platform == "cpu":
            dims = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray((num_rows, num_cols, batchsize, max_num_spikes), dtype=np.uint64))
            shape_dims = xla_client.Shape.array_shape(np.dtype(np.uint64), (2,), (0,))
            op_ = [xla_client.ops.CustomCallWithLayout(
                builder=ctx.builder,
                call_target_name=op_name.encode(),
                operands=(dims, *operands_base),
                shape_with_layout=matrix_grad_shape,
                operand_shapes_with_layout=(
                    shape_dims,
                    operand_shapes,
                ),
            )]
        elif platform == "gpu":
            opaque = struct.pack("IIII", num_rows, num_cols, batchsize, max_num_spikes)
            op_ = [xla_client.ops.CustomCallWithLayout(
                builder=ctx.builder,
                call_target_name=op_name.encode(),
                operands=operands_base,
                shape_with_layout=matrix_grad_shape,
                operand_shapes_with_layout=operand_shapes,
                opaque=opaque,
            )]
        else:
            raise NotImplementedError(f"Unsupported platform {platform}")
        return op_

    def _spike_vector_matmul_matrix_grad_batch(vector_arg_values, batch_axes, *, num_rows):
        """Computes the batched version of the primitive.
        
        This must be a JAX-traceable function.
        
        Since the multiply_add primitive already operates pointwise on arbitrary
        dimension tensors, to batch it we can use the primitive itself. This works as
        long as both the inputs have the same dimensions and are batched along the
        same axes. The result is batched along the axis that the inputs are batched.
        
        Args:
            vector_arg_values: a tuple of two arguments, each being a tensor of matching
            shape.
            batch_axes: the axes that are being batched. See vmap documentation.
        Returns:
            a tuple of the result, and the result axis that was batched. 
        """
        spike_vector, result_grad = vector_arg_values
        # if batch_axes[0] != 0:
        #     raise ValueError("Batching over matrix with other than `batch_axis=0` is not supported yet.")
        assert spike_vector.num_neurons == num_rows

        def prepare_result_grad_batched(result_grad, batch_axis):
            if batch_axis != 0:
                print("WARING: Choosing batch axis other than zero for result, will lead to memory copying and therefore to reduced performance.")
            result_grad = vector_arg_values[1]
            result_grad = jnp.moveaxis(result_grad, batch_axis, 0)
            # make sure array is contigous
            result_grad_shape = result_grad.shape
            result_grad = result_grad.ravel().reshape(*result_grad_shape) # TODO make sure this doens't copy if it already is contigous
            return result_grad

        if batch_axes[0] == 0:
            if batch_axes[1] == None:
                raise ValueError("Batching over spike_vector but not result_grad is not supported yet.")
            # TODO shoud check for ndim == 2 otherwise reshape both args to 2d
            result_grad = prepare_result_grad_batched(vector_arg_values[1], batch_axes[1])
            res = spike_vector_matmul_matrix_grad(spike_vector, result_grad), 0
        elif batch_axes[0] is None:
            batchsize = result_grad.shape[batch_axes[1]]
            result_grad = prepare_result_grad_batched(vector_arg_values[1], batch_axes[1])
            res = jnp.stack([spike_vector_matmul_matrix_grad(spike_vector, result_grad[i]) for i in range(batchsize)]), 0
        else:
            raise ValueError("Batching over spike_vector with other than `batch_axis=0` or `None` is not supported yet.")
        return res

    fn_capsule = create_pycapsule(so_file, fn_name)
    xla_client.register_custom_call_target(op_name.encode(), fn_capsule, platform)

    _spike_vector_matmul_matrix_grad_p = core.Primitive(op_name)  # Create the primitive
    _spike_vector_matmul_matrix_grad_p.def_impl(ft.partial(xla.apply_primitive, _spike_vector_matmul_matrix_grad_p))
    _spike_vector_matmul_matrix_grad_p.def_abstract_eval(_spike_vector_matmul_matrix_grad_abstract_eval)
    xla.register_translation(_spike_vector_matmul_matrix_grad_p, _spike_vector_matmul_matrix_grad_xla_translation, platform=platform)
    # ad.primitive_jvps[_spike_vector_matmul_matrix_grad_p] = _spike_vector_matmul_matrix_grad_value_and_jvp
    # ad.primitive_transposes[_spike_vector_matmul_matrix_grad_p] = _spike_vector_matmul_matrix_grad_transpose
    batching.primitive_batchers[_spike_vector_matmul_matrix_grad_p] = _spike_vector_matmul_matrix_grad_batch

    return spike_vector_matmul_matrix_grad