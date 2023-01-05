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

from interface_utils import create_pycapsule

def get_spike_vector_matmul_vector_grad_fn(op_name: str, so_file: str, fn_name: str, platform: str):

    def spike_vector_matmul_vector_grad(matrix, vals, spike_vector, *, use_grad_num_spikes=False):
        """The JAX-traceable way to use the JAX primitive.
        
        Note that the traced arguments must be passed as positional arguments
        to `bind`. 
        """
        return _spike_vector_matmul_vector_grad_p.bind(matrix, vals, spike_vector, use_grad_num_spikes=use_grad_num_spikes)

    def _spike_vector_matmul_vector_grad_abstract_eval(matrix, vals, spike_vector, *, use_grad_num_spikes):
        assert matrix.ndim == 2
        assert matrix.dtype == np.dtype("float32")
        print(vals.shape)
        print(spike_vector)
        if (spike_vector.batchsize > 1):
            assert len(vals.shape) == 2 
            assert vals.shape[0] >= spike_vector.batchsize
        #     assert vals.shape[1] >= spike_vector.max_num_spikes
        # else:
        #     assert vals.shape[-1] >= spike_vector.max_num_spikes 
        # print(vals.shape)
        # print(spike_vector)
        # print(spike_vector.shape)
        # print(spike_vector.spike_ids_shape)
        # sys.exit()
        batched = len(vals.shape) > 1

        # assert vals.shape == spike_vector.spike_ids_shape
        assert vals.dtype == np.dtype("float32")
        num_rows = matrix.shape[0]
        num_cols = matrix.shape[1]

        return spike_vector.update()

    def _spike_vector_matmul_vector_grad_xla_translation(ctx, avals_in, avals_out, matrix, vals, spike_vector, *, use_grad_num_spikes):
        matrix_shape = avals_in[0]
        vals_shape = avals_in[1]
        spike_vector_shape = avals_in[2]
        out_shape = avals_out[0]

        # Extract the dtype and shape
        dtype = matrix_shape.dtype
        num_rows, num_cols = matrix_shape.shape
        batchsize, max_num_spikes = spike_vector_shape.batchsize, spike_vector_shape.max_num_spikes

        # print(matrix_shape)
        # print(vals_shape)
        # print(spike_vector_shape)
        # sys.exit()



        operands_base = (matrix, vals, spike_vector)
        operand_shapes = [
            xla_client.Shape.array_shape(arr.dtype, arr.shape, tuple(range(len(arr.shape) - 1, -1, -1))) 
            for arr in avals_in
        ]
        out_shape_with_layout = xla_client.Shape.array_shape(out_shape.dtype, out_shape.shape, tuple(range(len(out_shape.shape) - 1, -1, -1)))

        if platform == "cpu":
            raise NotImplementedError(f"Unsupported platform {platform}")
            # dims = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray((num_cols, batchsize, max_num_spikes), dtype=np.uint64))
            # shape_dims = xla_client.Shape.array_shape(np.dtype(np.uint64), (2,), (0,))
            # op_ = [xla_client.ops.CustomCallWithLayout(
            #     builder=ctx.builder,
            #     call_target_name=op_name.encode(),
            #     operands=(dims, *operands_base),
            #     shape_with_layout=out_shape_with_layout,
            #     operand_shapes_with_layout=(
            #         shape_dims,
            #         operand_shapes,
            #     ),
            # )]
        elif platform == "gpu":
            opaque = struct.pack("III", num_cols, batchsize, max_num_spikes)
            op_ = [xla_client.ops.CustomCallWithLayout(
                builder=ctx.builder,
                call_target_name=op_name.encode(),
                operands=operands_base,
                shape_with_layout=out_shape_with_layout,
                operand_shapes_with_layout=operand_shapes,
                opaque=opaque,
            )]
        else:
            raise NotImplementedError(f"Unsupported platform {platform}")
        return op_

    def _spike_vector_matmul_vector_grad_batch(vector_arg_values, batch_axes, *, use_grad_num_spikes):
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
        matrix = vector_arg_values[0]
        result_grads = vector_arg_values[1]
        spike_vector = vector_arg_values[2]
        if batch_axes[0] != None:
            raise ValueError("Batching over matrix is not supported yet.")
        if batch_axes == (None, 0, None):
            assert spike_vector.batchsize == 1
            batchsize = result_grads.shape[0] 
            res = jnp.stack([spike_vector_matmul_vector_grad(matrix, result_grads[0], spike_vector) for i in range(batchsize)])
        elif batch_axes == (None, 0, 0):
            assert len(result_grads.shape) == 2
            assert spike_vector.batchsize == result_grads.shape[0]
            res = spike_vector_matmul_vector_grad(matrix, result_grads, spike_vector)
        else:
            raise ValueError("Requested `batch_axes={batch_axes}` not supported yet.")
        return res, 0

    fn_capsule = create_pycapsule(so_file, fn_name)
    xla_client.register_custom_call_target(op_name.encode(), fn_capsule, platform)

    _spike_vector_matmul_vector_grad_p = core.Primitive(op_name)  # Create the primitive
    _spike_vector_matmul_vector_grad_p.def_impl(ft.partial(xla.apply_primitive, _spike_vector_matmul_vector_grad_p))
    _spike_vector_matmul_vector_grad_p.def_abstract_eval(_spike_vector_matmul_vector_grad_abstract_eval)
    xla.register_translation(_spike_vector_matmul_vector_grad_p, _spike_vector_matmul_vector_grad_xla_translation, platform=platform)
    # ad.primitive_jvps[_spike_vector_matmul_vector_grad_p] = _spike_vector_matmul_vector_grad_value_and_jvp
    # ad.primitive_transposes[_spike_vector_matmul_vector_grad_p] = _spike_vector_matmul_vector_grad_transpose
    batching.primitive_batchers[_spike_vector_matmul_vector_grad_p] = _spike_vector_matmul_vector_grad_batch

    return spike_vector_matmul_vector_grad