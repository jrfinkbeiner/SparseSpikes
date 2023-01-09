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

def get_sparse_vector_matmul_fn(op_name: str, so_file: str, fn_name: str, platform: str, spike_vector_grad_fn):

    def sparse_vector_matmul(matrix, vals, spike_vector, *, use_grad_num_spikes=False):
        """The JAX-traceable way to use the JAX primitive.
        
        Note that the traced arguments must be passed as positional arguments
        to `bind`. 
        """
        return _sparse_vector_matmul_p.bind(matrix, vals, spike_vector, use_grad_num_spikes=use_grad_num_spikes)

    def _sparse_vector_matmul_abstract_eval(matrix, vals, spike_vector, *, use_grad_num_spikes):
        assert matrix.ndim == 2
        assert matrix.dtype == np.dtype("float32")
        if (spike_vector.batchsize > 1):
            
            # print("\nvals.shape", vals.shape)
            # TODO adjust for stacked...
            # assert len(vals.shape) == 2 
            
            assert vals.shape[-2] >= spike_vector.batchsize
            assert vals.shape[-1] >= spike_vector.max_num_spikes
        else:
            assert vals.shape[-1] >= spike_vector.max_num_spikes 
        # print(vals.shape)
        # print(spike_vector)
        # print(spike_vector.shape)
        # print(spike_vector.spike_ids_shape)
        # sys.exit()

        # assert vals.shape == spike_vector.spike_ids_shape
        assert vals.dtype == np.dtype("float32")
        num_rows = matrix.shape[0]
        num_cols = matrix.shape[1]

        # assert spike_vector.dtype == np.dtype("uint32")
        out_shape = (spike_vector.batchsize, num_cols) if spike_vector.batched else (num_cols,)
        return abstract_arrays.ShapedArray(out_shape, matrix.dtype)

    def _sparse_vector_matmul_xla_translation(ctx, avals_in, avals_out, matrix, vals, spike_vector, *, use_grad_num_spikes):
        matrix_shape = avals_in[0]
        vals_shape = avals_in[1]
        sparse_vector_shape = avals_in[2]
        out_shape = avals_out[0]

        # Extract the dtype and shape
        dtype = matrix_shape.dtype
        num_rows, num_cols = matrix_shape.shape
        batchsize, max_num_spikes = sparse_vector_shape.batchsize, sparse_vector_shape.max_num_spikes
        
        operands_base = (matrix, vals, spike_vector)
        operand_shapes = [
            xla_client.Shape.array_shape(arr.dtype, arr.shape, tuple(range(len(arr.shape) - 1, -1, -1))) 
            for arr in avals_in
        ]
        out_shape = xla_client.Shape.array_shape(out_shape.dtype, out_shape.shape, tuple(range(len(out_shape.shape) - 1, -1, -1)))

        if platform == "cpu":
            dims = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray((batchsize, num_cols, max_num_spikes, use_grad_num_spikes), dtype=np.uint64))
            shape_dims = xla_client.Shape.array_shape(np.dtype(np.uint64), (2,), (0,))
            op_ = [xla_client.ops.CustomCallWithLayout(
                builder=ctx.builder,
                call_target_name=op_name.encode(),
                operands=(dims, *operands_base),
                shape_with_layout=out_shape,
                operand_shapes_with_layout=(
                    shape_dims,
                    operand_shapes,
                ),
            )]
        elif platform == "gpu":
            opaque = struct.pack("IIII", batchsize, num_cols, max_num_spikes, use_grad_num_spikes)
            op_ = [xla_client.ops.CustomCallWithLayout(
                builder=ctx.builder,
                call_target_name=op_name.encode(),
                operands=operands_base,
                shape_with_layout=out_shape,
                operand_shapes_with_layout=operand_shapes,
                opaque=opaque,
            )]
        else:
            raise NotImplementedError(f"Unsupported platform {platform}")
        return op_

    # def _sparse_vector_matmul_value_and_jvp(arg_values, arg_tangents, *, use_grad_num_spikes):
    #     """Evaluates the primal output and the tangents (Jacobian-vector product).

    #     Given values of the arguments and perturbation of the arguments (tangents), 
    #     compute the output of the primitive and the perturbation of the output.

    #     This method must be JAX-traceable. JAX may invoke it with abstract values 
    #     for the arguments and tangents.

    #     Args:
    #         arg_values: a tuple of arguments
    #         arg_tangents: a tuple with the tangents of the arguments. The tuple has 
    #         the same length as the arg_values. Some of the tangents may also be the 
    #         special value ad.Zero to specify a zero tangent.
    #     Returns:
    #         a pair of the primal output and the tangent.
    #     """
    #     matrix, spike_vector = arg_values
    #     matrix_t, sparse_vector_t = arg_tangents        

    #     # Now we have a JAX-traceable computation of the output. 
    #     primal_out = sparse_vector_matmul(matrix, spike_vector, use_grad_num_spikes=use_grad_num_spikes)
        
    #     def make_zero(tan, likearr):
    #         return jax.lax.zeros_like_array(likearr) if type(tan) is ad.Zero else tan  
    
    #     matrix_t_z = make_zero(matrix_t, matrix)
    #     sparse_vector_t_z = make_zero(sparse_vector_t, spike_vector)

    #     # TODO this is not supposed to create the correct result but just to force jax to use the transpose rule
    #     # TODO is this correct when using num_spikes for grad or forward ?!
    #     tan0 = sparse_vector_matmul(matrix_t_z, spike_vector, use_grad_num_spikes=True)
    #     tan1 = sparse_vector_t_z*sparse_vector_t_z # sparse_vector_matmul(matrix, spike_vector) # TODO this does not work
    #     output_tangent = tan0 + tan1
    #     return (primal_out, output_tangent)

    def _sparse_vector_matmul_transpose(result_t, matrix, vals, spike_vector, *, use_grad_num_spikes):
        """Evaluates the transpose of a linear primitive.

        This method is only used when computing the backward gradient following 
        value_and_jvp, and is only needed for primitives that are used in the JVP 
        calculation for some other primitive. We need transposition for multiply_add_prim, 
        because we have used multiply_add_prim in the computation of the output_tangent in 
        multiply_add_value_and_jvp.

        In our case, multiply_add is not a linear primitive. However, it is used linearly 
        w.r.t. tangents in multiply_add_value_and_jvp:
            output_tangent(xt, yt, zt) = multiply_add_prim(xt, y, multiply_add_prim(x, yt, zt))
        
        Always one of the first two multiplicative arguments is a constant.

        Args:
            ct: the cotangent of the output of the primitive.
            x, y, z: values of the arguments. The arguments that are used linearly
                get an ad.UndefinedPrimal value. The other arguments get a constant
                value.
        Returns:
            a tuple with the cotangent of the inputs, with the value None
            corresponding to the constant arguments.
        """

        def calc_mat_grad():
            # matrix_grad = ad.Zero(matrix.aval) if type(result_t) is ad.Zero else matrix_grad_fn(spike_vector, result_t)
            # return matrix_grad
            return ad.Zero(matrix.aval)

        def calc_spike_vector_grad():
            sparse_vector_grad = ad.Zero(spike_vector.aval) if type(result_t) is ad.Zero else spike_vector_grad_fn(matrix, result_t, spike_vector)
            print("\ncalc_spike_vector_grad\n")
            # sys.exit()
            # return sparse_vector_grad
            return sparse_vector_grad

        def calc_vals_grad():
            # sparse_vector_grad = ad.Zero(spike_vector.aval) if type(result_t) is ad.Zero else sparse_vector_grad_fn(matrix, result_t)
            # return sparse_vector_grad
            return ad.Zero(vals.aval)

        if not ((not ad.is_undefined_primal(matrix)) and (ad.is_undefined_primal(vals)) and (not ad.is_undefined_primal(spike_vector))):
            raise NotImplementedError("Not implemented yet. Only gradients w.r.t. vals are supported.")

        # TODO is this correct for sparse matmul or just to force jax to use the transpose rule for the spike_matmul ?
        res = None, calc_spike_vector_grad(), None
        return res

    def _sparse_vector_matmul_batch(vector_arg_values, batch_axes, *, use_grad_num_spikes):
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
        print(batch_axes)
        if batch_axes[0] != None:
            raise ValueError("Batching over matrix is not supported yet.")
        if batch_axes[1] != 0:
            raise ValueError("Batching vals for other than `batch_axis=0` is not supported yet.")
        if batch_axes[2] != 0:
            raise ValueError("Batching spike_vector for other than `batch_axis=0` is not supported yet.")
        return sparse_vector_matmul(*vector_arg_values), 0


    fn_capsule = create_pycapsule(so_file, fn_name)
    xla_client.register_custom_call_target(op_name.encode(), fn_capsule, platform)

    _sparse_vector_matmul_p = core.Primitive(op_name)  # Create the primitive
    _sparse_vector_matmul_p.def_impl(ft.partial(xla.apply_primitive, _sparse_vector_matmul_p))
    _sparse_vector_matmul_p.def_abstract_eval(_sparse_vector_matmul_abstract_eval)
    xla.register_translation(_sparse_vector_matmul_p, _sparse_vector_matmul_xla_translation, platform=platform)
    # ad.primitive_jvps[_sparse_vector_matmul_p] = _sparse_vector_matmul_value_and_jvp
    ad.primitive_transposes[_sparse_vector_matmul_p] = _sparse_vector_matmul_transpose
    batching.primitive_batchers[_sparse_vector_matmul_p] = _sparse_vector_matmul_batch

    return sparse_vector_matmul