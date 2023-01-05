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
# from jax.lib import xla_client
# xla_client.register_custom_call_target(b"cpu_add", cpu_add_fn)

from xla_gen_spike_vector_grad import get_gen_spike_vector_grad_fn

# Helpful links:
# https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
# https://dfm.io/posts/extending-jax/
# https://github.com/dfm/extending-jax/blob/main/lib/cpu_ops.cc

def get_gen_spike_vector_fn(op_name: str, so_file: str, fn_name: str, platform: str):

    def gen_spike_vector(states, thresholds, max_num_spikes):
        """The JAX-traceable way to use the JAX primitive.
        
        Note that the traced arguments must be passed as positional arguments
        to `bind`. 
        """
        #   return _gen_spike_vector_p.bind(len(a), a, b)
        # pass static value as keyword argument as they are not traced and therefore handleed differently
        return _gen_spike_vector_p.bind(states, thresholds, max_num_spikes=max_num_spikes)

    def _gen_spike_vector_abstract_eval(states, thresholds, *, max_num_spikes):
        if len(states.shape) not in [1, 2]:
            raise ValueError("`states` must be a vector or a batch of vectors. Higher dimensional arrays are not supported yet.")
        if len(thresholds.shape) != 1:
            raise ValueError("Thresholds must be a vector of size 2. Variable thresholds per state are not supported yet.")
        batched = len(states.shape) == 2
        batchsize = states.shape[0] if batched else 1
        assert states.dtype == np.dtype("float32")
        assert thresholds.dtype == np.dtype("float32")
        assert type(max_num_spikes) == int
        # return AbstractSparseSpikeVector(batchsize if batched else None, max_num_spikes, states.shape[-1], np.dtype("uint32"))
        return AbstractSparseSpikeVector(batchsize if batched else None, max_num_spikes, states.shape[-1], None, np.dtype("float32"))
        # add_batch_dim = max(int((2 * batchsize) / max_num_spikes), 1)
        # c_shape = (batchsize + add_batch_dim, max_num_spikes)
        # return abstract_arrays.ShapedArray(c_shape, np.dtype("uint32"))

    def _gen_spike_vector_xla_translation(ctx, avals_in, avals_out, ac, bc, *, max_num_spikes):
    # def _gen_spike_vector_xla_translation(ctx, avals_in, avals_out, ac, bc):
        # The inputs have "shapes" that provide both the shape and the dtype

        ac_shape = avals_in[0]
        bc_shape = avals_in[1]
        cc_shape = avals_out[0]

        # Extract the dtype and shape
        dtype = ac_shape.dtype
        dims = ac_shape.shape
        assert len(dims) in [1, 2]
        assert bc_shape.dtype == dtype
        assert len(bc_shape.shape) == 1
        assert bc_shape.shape[0] == 2

        batched = len(dims) > 1
        batchsize = dims[0] if batched else 1
        num_neurons = dims[-1]
        max_num_spikes = cc_shape.sparse_shape[-1]

        # print(cc_shape)
        # print(cc_shape.sparse_shape)
        # print(cc_shape.shape)
        # print(cc_shape.dense_shape)
        # sys.exit()

        add_batch_dim = max(int((2 * batchsize) / max_num_spikes), 1)
        # assert cc_shape.dtype == np.uint32
        # print(type(cc_shape))
        # sys.exit()
        assert cc_shape.sparse_shape[-2:] == (2*batchsize + add_batch_dim, max_num_spikes)


        # # The total size of the input is the product across dimensions
        # dim0 = xla_client.ops.ConstantLiteral(ctx, np.asarray(ac_shape.shape[0]).sum().astype(np.int64))
        # dim1 = xla_client.ops.ConstantLiteral(ctx, np.asarray(ac_shape.shape[1]).sum().astype(np.int64))

        # print(condc)
        # print(dim0)
        # sys.exit()

        # val = xla_client.LiteralSlice(4)
        # sys.exit()

        # dim0 = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray(ac_shape.shape[0]).sum().astype(np.int64))
        # dim1 = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray(ac_shape.shape[1]).sum().astype(np.int64))
        
        dims = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray((batchsize, num_neurons, max_num_spikes), dtype=np.uint64))

        # # The inputs and outputs all have the same shape so let's predefine this
        # # specification
        # shape_dim0 = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())
        # shape_dim1 = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())
        shape_dims = xla_client.Shape.array_shape(np.dtype(np.uint64), (2,), (0,))

        shape_a = xla_client.Shape.array_shape(
            np.dtype(ac_shape.dtype), ac_shape.shape, tuple(range(len(ac_shape.shape) - 1, -1, -1))
        )
        shape_b = xla_client.Shape.array_shape(
            np.dtype(bc_shape.dtype), bc_shape.shape, tuple(range(len(bc_shape.shape) - 1, -1, -1))
        )
        
        shape_comb_spike_data = xla_client.Shape.array_shape(
            np.dtype(cc_shape.dtype), cc_shape.sparse_shape, tuple(range(len(cc_shape.sparse_shape) - 1, -1, -1))
        )


        # We dispatch a different call depending on the dtype
        # if dtype == np.float32:
        #     op_name = op_name
        # elif dtype == np.float64:
        #     op_name = platform.encode() + b"_kepler_f64"
        # else:
        #     raise NotImplementedError(f"Unsupported dtype {dtype}")

        # We pass the size of the data as a the first input argument
        
        # return [xla_client.ops.CustomCallWithLayout(
        #     builder=ctx.builder,
        #     call_target_name=op_name,
        #     # operands=(dims, condc, rowidc, colidc, bc),
        #     operands=(dims, ac, bc),
        #     shape_with_layout=shape_c,
        #     operand_shapes_with_layout=(
        #         # shape_dim0,
        #         # shape_dim1,

        #         shape_dims,
        #         # xla_client.Shape.tuple_shape(
        #         shape_cond,
        #         shape_rowids,
        #         shape_colids,
        #         shape_b,
        #     ),
        # )]

        if platform == "cpu":
            op_ = [xla_client.ops.CustomCallWithLayout(
                builder=ctx.builder,
                call_target_name=op_name.encode(),
                operands=(dims, ac, bc),
                # operands=(dims, ac, bc),
                shape_with_layout=shape_comb_spike_data,
                operand_shapes_with_layout=(
                    # shape_dim0,
                    # shape_dim1,

                    shape_dims,
                    # xla_client.Shape.tuple_shape((
                    #     shape_cond,
                    #     shape_rowids,
                    #     shape_colids,
                    # )),
                    shape_a,
                    shape_b,
                ),
            )]
        elif platform == "gpu":
            # opaque = f"{batchsize:32}{num_neurons:32}{max_num_spikes:32}".encode()
            
            opaque = struct.pack("III", batchsize, num_neurons, max_num_spikes)
            op_ = [xla_client.ops.CustomCallWithLayout(
                builder=ctx.builder,
                call_target_name=op_name.encode(),
                operands=(ac, bc),
                # operands=(dims, ac, bc),
                shape_with_layout=shape_comb_spike_data,
                operand_shapes_with_layout=(
                    shape_a,
                    shape_b,
                ),
                # opaque=bytes(dims.to_py()),
                opaque=opaque,
            )]
        else:
            raise NotImplementedError(f"Unsupported platform {platform}")
        
        return op_

    def _gen_spike_vector_value_and_jvp(arg_values, arg_tangents, *, max_num_spikes):
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
        states, thresholds = arg_values
        states_t, thresholds_t = arg_tangents

        # if not isinstance(thresholds_t, ad.Zero):
        #     raise NotImplementedError("Gradient computation with respect to threholds is not yet implemented.")
        # if isinstance(states_t, ad.Zero):
        #     raise ValueError("states_t is ad.Zero")

        # Now we have a JAX-traceable computation of the output. 
        # Normally, we can use the ma primtive itself to compute the primal output. 
        primal_out = gen_spike_vector(states, thresholds, max_num_spikes=max_num_spikes)

        #   # We must use a JAX-traceable way to compute the tangent. It turns out that 
        #   # the output tangent can be computed as (xt * y + x * yt + zt),
        #   # which we can implement in a JAX-traceable way using the same "multiply_add_prim" primitive.
        
        # We do need to deal specially with Zero. Here we just turn it into a 
        # proper tensor of 0s (of the same shape as 'x'). 
        # An alternative would be to check for Zero and perform algebraic 
        # simplification of the output tangent computation.

        def make_zero(tan, likearr):
            return jax.lax.zeros_like_array(likearr) if type(tan) is ad.Zero else tan  
    
        states_t_z = make_zero(states_t, states)
        thresholds_t_z = make_zero(thresholds_t, thresholds)

        # TODO this is not supposed to create the correct result but just to force jax to use the transpose rule
        tan0 = gen_spike_vector(states_t_z, thresholds_t_z, max_num_spikes=max_num_spikes)
        output_tangent = tan0

        return (primal_out, output_tangent)

    def _gen_spike_vector_transpose(ct, a, b, *, max_num_spikes):
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

        # sys.exit()
        # for arg in [a, b]:
            # if type(arg) is ad.UndefinedPrimal:
        #         raise NotImplementedError("Backprop through undefined primal not implemented")

        # if ad.is_undefined_primal(b):
        #     raise NotImplementedError("Gradient calculation w.r.t. `thresholds` is not implemented.")


        state_grad = gen_spike_vector_grad_fn(ct, states_fwd_shape=a.aval.shape, states_fwd_dtype=a.aval.dtype)
        # state_grad = jnp.ones(a.aval.shape)
        res = state_grad, None
        # assert 
        # if not ad.is_undefined_primal(a):
        #     # This use of multiply_add is with a constant "x"
        #     assert ad.is_undefined_primal(b)
        #     # ct_b = ad.Zero(b.aval) if type(ct) is ad.Zero else gen_spike_vector(ct, a)
        #     ct_b = ad.Zero(b.aval) if type(ct) is ad.Zero else ct @ a # TODO implement _rmatmul !!! with transpose flag
        #     res = None, ct_b
        # else:
        #     # This use of multiply_add is with a constant "y"
        #     assert ad.is_undefined_primal(a)
        #     ct_x = ad.Zero(b.aval) if type(ct) is ad.Zero else jnp.outer(ct, b)
        #     res = ct_x, None
        return res

    def _gen_spike_vector_batch(vector_arg_values, batch_axes, *, max_num_spikes):
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
        if batch_axes[1] != None:
            raise ValueError("Batching of thresholds is not supported yet.")
        states = vector_arg_values[0]
        if batch_axes[0] != 0:
            print("WARING: Choosing batch axis other than one for state, will lead to memory copying and therefore to reduced performance.")
            states = jnp.moveaxis(states, batch_axes[0], 0)
        # make sure array is contigous
        states_shape = states.shape
        states = states.ravel().reshape(*states_shape)
        thresholds = vector_arg_values[1]
        return gen_spike_vector(states, thresholds, max_num_spikes=max_num_spikes), 0

    gen_spike_vector_grad_fn = get_gen_spike_vector_grad_fn(op_name+"_grad", so_file, fn_name+"_grad", platform)

    fn_capsule = create_pycapsule(so_file, fn_name)
    xla_client.register_custom_call_target(op_name.encode(), fn_capsule, platform)

    _gen_spike_vector_p = core.Primitive(op_name)  # Create the primitive
    _gen_spike_vector_p.def_impl(ft.partial(xla.apply_primitive, _gen_spike_vector_p))
    _gen_spike_vector_p.def_abstract_eval(_gen_spike_vector_abstract_eval)
    xla.register_translation(_gen_spike_vector_p, _gen_spike_vector_xla_translation, platform=platform)
    ad.primitive_jvps[_gen_spike_vector_p] = _gen_spike_vector_value_and_jvp
    ad.primitive_transposes[_gen_spike_vector_p] = _gen_spike_vector_transpose
    batching.primitive_batchers[_gen_spike_vector_p] = _gen_spike_vector_batch

    return gen_spike_vector