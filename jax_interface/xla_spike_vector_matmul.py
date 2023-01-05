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
from xla_spike_vector import AbstractSparseSpikeVector
from xla_spike_vector_matmul_matrix_grad import get_spike_vector_matmul_matrix_grad_fn
from xla_spike_vector_matmul_vector_grad import get_spike_vector_matmul_vector_grad_fn
from xla_sparse_vector_matmul import get_sparse_vector_matmul_fn

# from jax.lib import xla_client
# xla_client.register_custom_call_target(b"cpu_add", cpu_add_fn)

# Helpful links:
# https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
# https://dfm.io/posts/extending-jax/
# https://github.com/dfm/extending-jax/blob/main/lib/cpu_ops.cc

def get_spike_vector_matmul_fn(op_name: str, so_file: str, fn_name: str, platform: str):

    def spike_vector_matmul(matrix, spike_vector, *, use_grad_num_spikes=False):
        """The JAX-traceable way to use the JAX primitive.
        
        Note that the traced arguments must be passed as positional arguments
        to `bind`. 
        """
        return _spike_vector_matmul_p.bind(matrix, spike_vector, use_grad_num_spikes=use_grad_num_spikes)

    def _spike_vector_matmul_abstract_eval(matrix, spike_vector, *, use_grad_num_spikes):
        assert matrix.ndim == 2
        assert matrix.dtype == np.dtype("float32")
        assert isinstance(spike_vector, AbstractSparseSpikeVector)

        num_rows = matrix.shape[0]
        num_cols = matrix.shape[1]

        # assert spike_vector.dtype == np.dtype("uint32")
        out_shape = (spike_vector.batchsize, num_cols) if spike_vector.batched else (num_cols,)
        return abstract_arrays.ShapedArray(out_shape, matrix.dtype)

    def _spike_vector_matmul_xla_translation(ctx, avals_in, avals_out, matrix, spike_vector, *, use_grad_num_spikes):
        matrix_shape = avals_in[0]
        spike_vector_shape = avals_in[1]
        out_shape = avals_out[0]

        # Extract the dtype and shape
        dtype = matrix_shape.dtype
        num_rows, num_cols = matrix_shape.shape
        batchsize, max_num_spikes = spike_vector_shape.batchsize, spike_vector_shape.max_num_spikes
        
        operands_base = (matrix, spike_vector)
        operand_shapes = [
            xla_client.Shape.array_shape(arr.dtype, arr.shape, tuple(range(len(arr.shape) - 1, -1, -1))) 
            for arr in [matrix_shape, spike_vector_shape]
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

    def _spike_vector_matmul_value_and_jvp(arg_values, arg_tangents, *, use_grad_num_spikes):
        """Evaluates the primal output and the tangents (Jacobian-vector product).

        Given values of the arguments and perturbation of the arguments (tangents), 
        compute the output of the primitive and the perturbation of the output.

        This method must be JAX-traceable. JAX may invoke it with abstract values 
        for the arguments and tangents.

        Args:
            arg_values: a tuple of arguments
            arg_tangents: a tuple with the tangents of the arguments. The tuple has 
            the same length as the arg_values. Some of the tangents may also be the 
            special value ad.Zero to specify a zero tangent.
        Returns:
            a pair of the primal output and the tangent.
        """
        matrix, spike_vector = arg_values
        matrix_t, spike_vector_t = arg_tangents        

        # Now we have a JAX-traceable computation of the output. 
        primal_out = spike_vector_matmul(matrix, spike_vector, use_grad_num_spikes=use_grad_num_spikes)
        
        def make_zero(tan, likearr):
            return jax.lax.zeros_like_array(likearr) if type(tan) is ad.Zero else tan  
    
        matrix_t_z = make_zero(matrix_t, matrix)
        spike_vector_t_z = make_zero(spike_vector_t, spike_vector)

        # TODO this is not supposed to create the correct result but just to force jax to use the transpose rule
        # TODO is this correct when using num_spikes for grad or forward ?!
        tan0 = spike_vector_matmul(matrix_t_z, spike_vector, use_grad_num_spikes=False)
        # TODO rewrite the `sparse_vector_matmul_fn`
        tan1 = sparse_vector_matmul_fn(matrix, spike_vector_t_z, spike_vector, use_grad_num_spikes=True) # spike_vector_matmul(matrix, spike_vector) # TODO this does not work
        output_tangent = tan0 + tan1
        return (primal_out, output_tangent)

    def _spike_vector_matmul_transpose(result_t, matrix, spike_vector, *, use_grad_num_spikes):
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
            matrix_grad = ad.Zero(matrix.aval) if type(result_t) is ad.Zero else matrix_grad_fn(spike_vector, result_t)
            return matrix_grad

        def calc_spike_vector_grad():
            spike_vector_grad = ad.Zero(spike_vector.aval) if type(result_t) is ad.Zero else spike_vector_grad_fn(matrix, result_t, spike_vector)
            return spike_vector_grad

        if (not ad.is_undefined_primal(spike_vector)) and ad.is_undefined_primal(matrix):
            res = calc_mat_grad(), None
        elif (not ad.is_undefined_primal(matrix)) and ad.is_undefined_primal(spike_vector):
            res = None, calc_spike_vector_grad()
        else:
            res = calc_mat_grad(), calc_spike_vector_grad()
        return res

    def _spike_vector_matmul_batch(vector_arg_values, batch_axes, *, use_grad_num_spikes):
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
        if batch_axes[0] != None:
            raise ValueError("Batching over matrix is not supported yet.")
        if batch_axes[1] != 0:
            raise ValueError("Batching spike_vector is only supported for `batch_axis=0` is not supported yet.")
        return spike_vector_matmul(*vector_arg_values), 0


    matrix_grad_fn = get_spike_vector_matmul_matrix_grad_fn(op_name+"_matrix_grad", so_file, fn_name+"_matrix_grad", platform)
    spike_vector_grad_fn = get_spike_vector_matmul_vector_grad_fn(op_name+"_spikes_grad", so_file, fn_name+"_spikes_grad", platform)
    sparse_vector_matmul_fn = get_sparse_vector_matmul_fn("sparse_vector_matmul_gpu_f32", so_file, "sparse_vector_matmul_gpu_f32", platform, spike_vector_grad_fn=spike_vector_grad_fn)

    fn_capsule = create_pycapsule(so_file, fn_name)
    xla_client.register_custom_call_target(op_name.encode(), fn_capsule, platform)

    _spike_vector_matmul_p = core.Primitive(op_name)  # Create the primitive
    _spike_vector_matmul_p.def_impl(ft.partial(xla.apply_primitive, _spike_vector_matmul_p))
    _spike_vector_matmul_p.def_abstract_eval(_spike_vector_matmul_abstract_eval)
    xla.register_translation(_spike_vector_matmul_p, _spike_vector_matmul_xla_translation, platform=platform)
    ad.primitive_jvps[_spike_vector_matmul_p] = _spike_vector_matmul_value_and_jvp
    ad.primitive_transposes[_spike_vector_matmul_p] = _spike_vector_matmul_transpose
    batching.primitive_batchers[_spike_vector_matmul_p] = _spike_vector_matmul_batch

    return spike_vector_matmul