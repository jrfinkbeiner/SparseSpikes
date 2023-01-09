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
from .xla_spike_vector import AbstractSparseSpikeVector
# from jax.lib import xla_client
# xla_client.register_custom_call_target(b"cpu_add", cpu_add_fn)

# Helpful links:
# https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
# https://dfm.io/posts/extending-jax/
# https://github.com/dfm/extending-jax/blob/main/lib/cpu_ops.cc



def get_gen_spike_vector_grad_fn(op_name: str, so_file: str, fn_name: str, platform: str):

    def gen_spike_vector_grad(spike_vector, *, states_fwd_shape, states_fwd_dtype):
        """Generate a gradient for a spike vector generation function.

        Args:
            spike_vector (jax.interpreters.xla.DeviceArray): Forward pass spike vector with precalculated surrogate gradient.
        
        Returns:
            jax.interpreters.xla.DeviceArray: Gradient of the loss with respect to the states.
        """
        return _gen_spike_vector_grad_p.bind(spike_vector, states_fwd_shape=states_fwd_shape, states_fwd_dtype=states_fwd_dtype)

    def _gen_spike_vector_grad_abstract_eval(spike_vector, *, states_fwd_shape, states_fwd_dtype):
        if len(states_fwd_shape) not in [1, 2]:
            raise ValueError("`states` must be a vector or a batch of vectors. Higher dimensional arrays are not supported yet.")
        batched = len(states_fwd_shape) == 2
        assert batched == spike_vector.batched
        if batched:
            assert states_fwd_shape[0] == spike_vector.batchsize
        assert states_fwd_dtype == np.dtype("float32")
        return abstract_arrays.ShapedArray(states_fwd_shape, states_fwd_dtype)

    def _gen_spike_vector_grad_xla_translation(ctx, avals_in, avals_out, spike_vector, *, states_fwd_shape, states_fwd_dtype):
        spike_vector_c_shape = avals_in[0]
        states_grad_c_shape = avals_out[0]

        # Extract the dtype and shape
        dtype = states_fwd_dtype
        dims = states_fwd_shape
        batchsize = spike_vector_c_shape.batchsize
        num_neurons = dims[-1]
        max_num_spikes = spike_vector_c_shape.max_num_spikes
        if spike_vector_c_shape.batched:
            assert dims[0] == batchsize

        # # The inputs and outputs all have the same shape so let's predefine this
        # # specification
        # shape_dim0 = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())
        # shape_dim1 = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())
        shape_dims = xla_client.Shape.array_shape(np.dtype(np.uint64), (2,), (0,))

        # operand_shapes = [xla.aval_to_xla_shape(av) for av in avals_in]
        operand_shapes = [
            xla_client.Shape.array_shape(arr.dtype, arr.shape, tuple(range(len(arr.shape) - 1, -1, -1))) 
            for arr in [spike_vector_c_shape]
        ]
        out_shape = xla_client.Shape.array_shape(states_grad_c_shape.dtype, states_grad_c_shape.shape, tuple(range(len(states_grad_c_shape.shape) - 1, -1, -1)))

        if platform == "cpu":
            dims = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray((batchsize, num_neurons, max_num_spikes), dtype=np.uint64))
            # op_ = [xla_client.ops.CustomCallWithLayout(
            #     builder=ctx.builder,
            #     call_target_name=op_name.encode(),
            #     operands=(dims, states_fwd, thresholds_fwd, spike_vector_fwd, spike_grads),
            #     shape_with_layout=out_shape,
            #     operand_shapes_with_layout=(
            #         shape_dims,
            #         operand_shapes,
            #     ),
            # )]
            raise NotImplementedError("CPU not implemented yet.")
        elif platform == "gpu":
            # TODO now for precalculated surrogate gradients
            opaque = struct.pack("III", batchsize, num_neurons, max_num_spikes)
            op_ = [xla_client.ops.CustomCallWithLayout(
                builder=ctx.builder,
                call_target_name=op_name.encode(),
                # operands=(states_fwd, thresholds_fwd, spike_vector_fwd, spike_grads),
                operands=(spike_vector,),
                shape_with_layout=out_shape,
                operand_shapes_with_layout=operand_shapes,
                opaque=opaque,
            )]
        else:
            raise NotImplementedError(f"Unsupported platform {platform}")
        
        return op_

    def _gen_spike_vector_grad_batch(vector_arg_values, batch_axes, *, states_fwd_shape, states_fwd_dtype):
        """Computes the batched version of the primitive.
        
        Args:
            vector_arg_values: a tuple of two arguments, each being a tensor of matching
            shape.
            batch_axes: the axes that are being batched. See vmap documentation.
        Returns:
            a tuple of the result, and the result axis that was batched. 
        """
        assert batch_axes == (0,)
        return gen_spike_vector_grad(vector_arg_values), 0
    
    fn_capsule = create_pycapsule(so_file, fn_name)
    xla_client.register_custom_call_target(op_name.encode(), fn_capsule, platform)

    # print(cpu_add_f32_fn)
    # print(type(cpu_add_f32_fn))
    # print(cpu_add_f32_fn_capsule)
    # print(type(cpu_add_f32_fn_capsule))

    _gen_spike_vector_grad_p = core.Primitive(op_name)  # Create the primitive
    _gen_spike_vector_grad_p.def_impl(ft.partial(xla.apply_primitive, _gen_spike_vector_grad_p))
    _gen_spike_vector_grad_p.def_abstract_eval(_gen_spike_vector_grad_abstract_eval)
    xla.register_translation(_gen_spike_vector_grad_p, _gen_spike_vector_grad_xla_translation, platform=platform)
    # ad.primitive_jvps[_gen_spike_vector_grad_p] = _gen_spike_vector_grad_value_and_jvp
    # ad.primitive_transposes[_gen_spike_vector_grad_p] = _gen_spike_vector_grad_transpose
    batching.primitive_batchers[_gen_spike_vector_grad_p] = _gen_spike_vector_grad_batch

    return gen_spike_vector_grad