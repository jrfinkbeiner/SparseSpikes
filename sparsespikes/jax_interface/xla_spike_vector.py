import sys
from typing import Callable, Optional, Sequence
from dataclasses import dataclass
import numpy as np
import jax

from jax._src.core import aval_property
from jax import core, jit, lax, make_jaxpr
from jax._src import device_array
from jax._src import dispatch
from jax._src import dtypes
from jax._src import ad_util
from jax.interpreters import mlir
from jax.interpreters import xla
from jax._src.lib.mlir import ir
from jax._src.lib import xla_client
from jax._src import xla_bridge
xc = xla_client
xb = xla_bridge

from jax.config import config
config.parse_flags_with_absl()

# TODO(jakevdp): use a setup/teardown method to populate and unpopulate all the
# dictionaries associated with the following objects.

# Define a sparse array data structure. The important feature here is that
# it is a jaxpr object that is backed by two device buffers.
class SparseSpikeVector:
    """Data structure representing sparse spike vector."""
    def __init__(self, spike_ids=None, num_spikes=None, num_neurons=None, comb_spike_data=None, aval=None):
        # TODO corretly handle index_dtypes...
        if comb_spike_data is None:
            assert spike_ids is not None
            assert num_spikes is not None
            assert num_neurons is not None
            assert len(spike_ids.shape) in [1, 2, 3]
            assert len(num_spikes.shape) == len(spike_ids.shape)
            batched = len(spike_ids.shape) > 1
            batchsize = 1 if not batched else spike_ids.shape[-2]
            is_stacked = len(spike_ids.shape) > 2
            if is_stacked and (not batched):
                raise ValueError("Currently stacking is only supported for batched spike_ids")
                
            stack_size = 1 if not is_stacked else spike_ids.shape[-3]
            
            max_num_spikes = spike_ids.shape[-1]
                        
            spike_ids = spike_ids.astype(np.uint32) # TODO really astype or view?
            num_spikes = num_spikes.astype(np.uint32) # TODO really astype or view?

            if not batched: 
                spike_ids = spike_ids[None, :]
            if batched:
                assert num_spikes.shape[-1] == batchsize
            shape = AbstractSparseSpikeVector.calc_comb_spike_data_shape(is_stacked, batched, stack_size, batchsize, max_num_spikes)
            # comb_spike_data = jax.numpy.empty(shape, dtype=spike_ids.dtype).reshape((stack_size, -1))
            spike_grad_data = jax.numpy.empty_like(spike_ids)
            # comb_spike_data[:, :max_num_spikes*batchsize] = spike_ids.reshape((stack_size, -1))
            # comb_spike_data[:, -2*batchsize:] = num_spikes.reshape((stack_size, -1))
            # print()
            comb_spike_data = jax.numpy.concatenate([spike_ids.reshape((stack_size, -1)), spike_grad_data.reshape((stack_size, -1)), num_spikes.reshape((stack_size, -1))], axis=-1, dtype=np.uint32)
            comb_spike_data = comb_spike_data.reshape(shape).view(dtype=np.float32)
        else:
            assert aval is not None
            assert np.dtype(comb_spike_data.dtype) == np.float32
            batched = aval.batched

        if aval is None:
            aval = AbstractSparseSpikeVector(batchsize if batched else None, max_num_spikes, num_neurons, stack_size if is_stacked else None, np.float32)
        self.aval = aval
        self._comb_spike_data = comb_spike_data

    @property
    def shape(self):
        return self.aval.sparse_shape

    @property
    def num_neurons(self):
        return self.aval.num_neurons

    @property
    def batchsize(self):
        return self.aval.batchsize

    @property
    def stack_size(self):
        return self.aval.stack_size


    @property
    def max_num_spikes(self):
        return self.aval.max_num_spikes

    @property
    def dtype(self):
        return self.aval.dtype

    @property
    def ndim(self):
        return self.aval.ndim

    @property
    def comb_spike_data(self):
        # print(type(self._comb_spike_data))
        return self._comb_spike_data

    @property
    def spike_ids(self):
        nvals = self.batchsize*self.max_num_spikes 
        data = self.comb_spike_data.reshape((self.stack_size, -1))[:, :nvals]
        shape = (*self.shape[:-1], self.max_num_spikes)
        data = data.reshape(shape)
        return data.view(dtype=np.uint32)

    @property
    def spike_grads(self):
        nvals = self.batchsize*self.max_num_spikes 
        data = self.comb_spike_data.reshape((self.stack_size, -1))[:, nvals:2*nvals]
        shape = (*self.shape[:-1], self.max_num_spikes)
        data = data.reshape(shape)
        return data.view(dtype=np.float32)

    @property
    def size_num_spikes(self):
        return self.batchsize*2

    @property
    def batched(self):
        return self.aval.batched

    @property
    def is_stacked(self):
        return self.aval.is_stacked

    @property
    def num_spikes(self):
        nvals = self.batchsize*self.max_num_spikes 
        data = self.comb_spike_data.reshape((self.stack_size, -1))[:, 2*nvals:]
        shape = []
        if self.is_stacked:
            shape.append(self.stack_size)
        shape.append(2)
        if self.batched:
            shape.append(self.batchsize)
        data = data.reshape(shape)
        return data.view(dtype=np.uint32)

    def __repr__(self):
        # TODO include spike_grads here ?
        return f"SparseSpikeVector(spike_ids={self.spike_ids}, spike_grads={self.spike_grads}, num_spikes={self.num_spikes})"

    def __getitem__(self, idx):
        assert self.is_stacked
        aval = self.aval.update(stack_size=-1)
        return SparseSpikeVector(comb_spike_data=self.comb_spike_data[idx], aval=aval)

    def __iter__(self):
        assert self.is_statcked
        aval = self.aval.update(stack_size=-1)
        for i in range(self.stack_size):
            yield SparseSpikeVector(comb_spike_data=self.comb_spike_data[i], aval=aval)

    # def __matmul__(self, b):
    #     # print("\n__matmul__")
    #     # print("self", self)
    #     # print("b", b)
    #     # print()
    #     # return mcbmm(self, b)
    #     return self.aval._matmul(self, b)

    # def __add__(self, b):
    #     # print("\n__matmul__")
    #     # print("self", self)
    #     # print("b", b)
    #     # print()
    #     # return mcbadd(self, b)
    #     new_conductences = self.aval._add(self, b)
    #     return type(self)(new_conductences, row_ids=self.row_ids, col_ids=self.col_ids, aval=self.aval)

comb_spike_data_p = core.Primitive('comb_spike_data')

@comb_spike_data_p.def_impl
def _comb_spike_data_impl(mat):
  return mat.comb_spike_data_p

@comb_spike_data_p.def_abstract_eval
def _comb_spike_data_abstract_eval(mat):
  return mat.comb_spike_data_aval

# Note: cannot use lower_fun to define attribute access primitives
# because it leads to infinite recursion.

def _comb_spike_data_mhlo_lowering(ctx, comb_spike_data):
  return [comb_spike_data[0]]

mlir.register_lowering(comb_spike_data_p, _comb_spike_data_mhlo_lowering)

# class AbstractSparseSpikeVector(core.UnshapedArray):
class AbstractSparseSpikeVector(core.ShapedArray):
    __slots__ = ['sparse_shape', 'dense_shape', 'comb_spike_data_aval', 'is_stacked', 'stack_size', 'batched', 'batchsize', 'max_num_spikes', 'num_neurons']
    # __slots__ = ['sparse_shape', 'comb_spike_data_aval', 'is_stacked', 'stack_size', 'batched', 'batchsize', 'max_num_spikes', 'num_neurons']

    def __init__(self, batchsize, max_num_spikes, num_neurons, stack_size, dtype, weak_type=False) :
            # , named_shape=None):
        named_shape = None
        self.num_neurons = num_neurons
        self.is_stacked = True if stack_size is not None else False
        self.stack_size = stack_size if stack_size is not None else 1 
        self.batched = True if batchsize is not None else False
        self.batchsize = batchsize if batchsize is not None else 1
        self.max_num_spikes = max_num_spikes
        comb_data_shape = self.calc_comb_spike_data_shape(self.is_stacked, self.batched, self.stack_size, self.batchsize, max_num_spikes)
        super().__init__(comb_data_shape, dtypes.canonicalize_dtype(dtype))
        # super().__init__(dtypes.canonicalize_dtype(dtype))
        # # self.shape = self.calc_dense_spike_data_shape(self.stack_size, self.batchsize, num_neurons)
        # self.shape = comb_data_shape
        self.sparse_shape = comb_data_shape
        self.dense_shape = self.calc_dense_spike_data_shape(self.stack_size, self.batchsize, num_neurons)
        
        assert np.dtype(dtype) == np.float32

        assert len(comb_data_shape) in (1, 2, 3)
        named_shape = {} if named_shape is None else named_shape
        self.comb_spike_data_aval = core.ShapedArray(comb_data_shape, dtypes.canonicalize_dtype(dtype),
                                        weak_type, named_shape)

    @property
    def underlying_dtype(self):
        return np.uint32

    @staticmethod
    def calc_comb_spike_data_shape(is_stacked: bool, batched: bool, stack_size: int, batchsize: int, max_num_spikes: int):
        last_dim = 2*max_num_spikes + 2
        shape = []
        if is_stacked:
            shape.append(stack_size)
        if batched:
            shape.append(batchsize)
        shape.append(last_dim)    
        return tuple(shape)

    def calc_dense_spike_data_shape(self, stack_size: int, batchsize: int, num_neurons: int):
        dense_shape = []
        if self.is_stacked:
            dense_shape.append(stack_size)
        if self.batched:
            dense_shape.append(batchsize)
        dense_shape.append(num_neurons)
        return tuple(dense_shape)


    def update(self, batchsize=None, max_num_spikes=None, num_neurons=None, stack_size=None,
                dtype=None, weak_type=None, named_shape=None):
        if batchsize is None:
            batchsize = self.batchsize if self.batched else None
        elif batchsize == -1:
            batchsize = None
        if max_num_spikes is None:
            max_num_spikes = self.max_num_spikes
        if num_neurons is None:
            num_neurons = self.num_neurons
        if stack_size is None:
            stack_size = self.stack_size if self.is_stacked else None
        elif stack_size == -1:
            stack_size = None
        if dtype is None:
            dtype = self.dtype
        if weak_type is None:
            weak_type = self.weak_type
        if named_shape is None:
            named_shape = self.named_shape
        return type(self)(
            batchsize, max_num_spikes, num_neurons, stack_size, dtype, weak_type) #, named_shape)

    @property
    def spike_ids_shape(self):
        # TODO probably not working properly with vmap
        if self.batched:
            return (self.batchsize, self.max_num_spikes)
        else:
            return (self.max_num_spikes,)

    def strip_weak_type(self):
        return self

    @aval_property
    def comb_spike_data(self):
        return comb_spike_data_p.bind(self)

    def _add(self, a, b):
        raise NotImplementedError
        # return mcbadd(a, b)

    def _matmul(self, a, b): 
        raise NotImplementedError
        # return mcbmm(a, b)

    def _rmatmul(self, b): 
        raise NotImplementedError
        # return mcbmm(self, b)

    def __getitem__(self, idx):
        assert self.is_statcked
        # TODO use update when the None type issue is fixed
        return AbstractSparseSpikeVector(self.batchsize, self.max_num_spikes, self.num_neurons, None, self.dtype, self.weak_type, self.named_shape)

    def __iter__(self):
        assert self.is_statcked
        for i in range(self.stack_size):
            yield self[i]


def check_is_sparse_spikes_type(inp_spikes):
    is_sparse_spikes_type = False
    if isinstance(inp_spikes, (SparseSpikeVector, AbstractSparseSpikeVector)):
        is_sparse_spikes_type = True
    elif hasattr(inp_spikes, "aval"): # in case of tracer # TODO how to do this properly?
        if isinstance(inp_spikes.aval, (SparseSpikeVector, AbstractSparseSpikeVector)):
            is_sparse_spikes_type = True
    return is_sparse_spikes_type


def sparse_spike_vector_result_handler(device, aval):
    def build_sparse_spike_vector(_, comb_spike_data_buf):
        comb_spike_data = device_array.make_device_array(aval.comb_spike_data_aval, device, comb_spike_data_buf)
        return SparseSpikeVector(comb_spike_data=comb_spike_data, aval=aval)
    return build_sparse_spike_vector

def sparse_spike_vector_shape_handler(a):
    # print("sparse_spike_vector_shape_handler", a)
    # sys.exit()
    return (xc.Shape.array_shape(a.dtype, a.shape), )

    # return (
    #     xc.Shape.array_shape(a.dtype, a.shape),
    # )

    # return (xc.Shape.tuple_shape((
    #     xc.Shape.array_shape(a.conductences_aval.dtype, a.conductences_aval.shape),
    #     xc.Shape.array_shape(a.row_ids_aval.dtype, a.row_ids_aval.shape),
    #     xc.Shape.array_shape(a.col_ids_aval.dtype, a.col_ids_aval.shape),
    # )),)

def sparse_spike_vector_device_put_handler(a, device):
    return (*dispatch.device_put(a.comb_spike_data, device), )

core.pytype_aval_mappings[SparseSpikeVector] = lambda x: x.aval
core.raise_to_shaped_mappings[AbstractSparseSpikeVector] = lambda aval, _: aval
xla.pytype_aval_mappings[SparseSpikeVector] = lambda x: x.aval
xla.canonicalize_dtype_handlers[SparseSpikeVector] = lambda x: x
# dispatch.device_put_handlers[SparseSpikeVector] = sparse_spike_vector_device_put_handler
# dispatch.result_handlers[AbstractSparseSpikeVector] = sparse_spike_vector_result_handler
# dispatch.num_buffers_handlers[AbstractSparseSpikeVector] = lambda _: 1
xla.xla_shape_handlers[AbstractSparseSpikeVector] = sparse_spike_vector_shape_handler

def sparse_spike_vector_mlir_type_handler(a):
  return (
    ir.RankedTensorType.get(
          a.shape, mlir.dtype_to_ir_type(a.dtype)),
  )
mlir.ir_type_handlers[AbstractSparseSpikeVector] = sparse_spike_vector_mlir_type_handler # TODO change this guy for single array handling in add ?

def _map_shaped_array(
    size: int, axis: Optional[int], aval: AbstractSparseSpikeVector) -> AbstractSparseSpikeVector:
  assert axis is None or aval.shape[axis] == size

#   print("\n_map_shaped_array", aval, size, axis )
  # print(aval)
  if axis==0:
    # assert aval.is_stacked # TODO make this mandatory?
    retval = aval.update(stack_size=-1)
  elif axis==1:
    assert aval.batched
    retval = aval.update(batchsize=-1)
  elif axis is None:
    retval = aval
  else:
    raise NotImplementedError
  return retval

def _unmap_shaped_array(
    size: int, axis_name, axis: Optional[int], aval: AbstractSparseSpikeVector
  ) -> AbstractSparseSpikeVector:
  named_shape = dict(aval.named_shape)
  named_shape.pop(axis_name, None)  # TODO: make this mandatory?
#   print("\n_unmap_shaped_array", aval, size, axis, named_shape)

  if axis is None: 
    ret = aval.update(named_shape=named_shape)
  elif type(axis) is int:
    if not aval.batched:
      assert axis==0
      ret = aval.update(batchsize=size, named_shape=named_shape)
    elif not aval.is_stacked:
      assert axis==0
      ret = aval.update(stack_size=size, named_shape=named_shape)
    else: 
      ValueError("Only possible to unmap `AbstractSparseSpikeVector` if it is not stacked or batched already.")
  else: raise TypeError(axis)
  return ret

core.aval_mapping_handlers[AbstractSparseSpikeVector] = (_map_shaped_array, _unmap_shaped_array)


# to make `jax.lax.scan` work
import operator
import functools as ft
from jax._src.lax.utils import _max, standard_abstract_eval, _standard_weak_type_rule, standard_named_shape_rule

def spike_and_standard_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule,
                           named_shape_rule, *avals, **kwargs):
  assert all(isinstance(aval, core.UnshapedArray) for aval in avals), avals
  assert not prim.multiple_results
  weak_type = weak_type_rule(*avals, **kwargs)
  least_specialized = _max(map(type, avals),
                           key=operator.attrgetter('array_abstraction_level'))
  
  if least_specialized is AbstractSparseSpikeVector:
    shape = shape_rule(*avals, **kwargs)
    dtype = dtype_rule(*avals, **kwargs)
    named_shape = named_shape_rule(*avals, **kwargs)
    assert isinstance(avals[0], AbstractSparseSpikeVector)
    stack_size = shape[0] if len(shape)==3 else -1
    return avals[0].update(stack_size=stack_size, dtype=dtype, weak_type=weak_type, named_shape=named_shape)
    # return AbstractSparseSpikeVector(batchsize, max_num_spikes, num_neurons, stack_size, dtype=dtype, weak_type=weak_type, named_shape=named_shape)
  else:
    return standard_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule,
                           named_shape_rule, *avals, **kwargs)

from jax._src.lax.slicing import dynamic_slice_p, _dynamic_slice_shape_rule, _dynamic_slice_dtype_rule, _argnum_weak_type

dynamic_slice_p.def_abstract_eval(
      ft.partial(spike_and_standard_abstract_eval, dynamic_slice_p, _dynamic_slice_shape_rule, _dynamic_slice_dtype_rule,
              _argnum_weak_type(0), standard_named_shape_rule))

from jax._src.lax.lax import squeeze_p, _squeeze_shape_rule, _squeeze_dtype_rule

squeeze_p.def_abstract_eval(
      ft.partial(spike_and_standard_abstract_eval, squeeze_p, _squeeze_shape_rule, _squeeze_dtype_rule,
              _standard_weak_type_rule, standard_named_shape_rule))

from jax._src.ad_util import aval_zeros_likers, jaxval_zeros_likers

def _zeros_like_sparse_spike_vector(aval):
  return SparseSpikeVector(comb_spike_data=np.zeros(aval.shape, np.uint32).view(aval.dtype), aval=aval)

aval_zeros_likers[AbstractSparseSpikeVector] = _zeros_like_sparse_spike_vector
# weak_type


# from jax._src.lib.mlir.dialects import mhlo

# def _sparse_spike_vector_constant(x: SparseSpikeVector, canonicalize_types
#                          ) -> Sequence[ir.Value]:
# #   print(x)
#   x = x.comb_spike_data
# #   sys.exit()
#   if canonicalize_types:
#     x = np.asarray(x, dtypes.canonicalize_dtype(x.dtype))

#   element_type = mlir.dtype_to_ir_type(x.dtype)
#   shape = x.shape
#   if x.dtype == np.bool_:
#     nelems = x.size
#     x = np.packbits(x, bitorder='little')
#     # TODO(b/209005197): Work around for MLIR crash for non-splat single element
#     # buffers.
#     if nelems == 1:
#       x = np.array(0 if x.item() == 0 else 0xff, np.uint8)
#   elif x.dtype == dtypes.bfloat16:
#     x = x.view(np.uint16)
#   x = np.ascontiguousarray(x)
#   attr = ir.DenseElementsAttr.get(x, type=element_type, shape=shape)
#   return (mhlo.ConstantOp(attr).result,)

# def _sparse_spike_vector_constant_handler(val: SparseSpikeVector, canonicalize_types
#                              ) -> Sequence[ir.Value]:
#   """Constant handler for ndarray literals, handling zero-size strides.

#   In most cases this function calls _numpy_array_constant(val) except it has
#   special handling of arrays with any strides of size zero: for those, it
#   generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
#   to avoid staging in large literals that might arise from np.zeros or np.ones
#   or the output of lax.broadcast (which uses np.broadcast_to which in turn
#   uses size-zero strides).

#   Args:
#     val: an ndarray.

#   Returns:
#     An XLA ComputationDataHandle / XlaOp representing the constant ndarray
#     staged into the XLA Computation.
#   """
#   if dtypes.result_type(val) == dtypes.float0:
#     raise NotImplementedError("float0")
#     return _numpy_array_constant(np.zeros(val.shape, dtype=np.bool_),
#                                 canonicalize_types=False)
# #   elif np.any(np.equal(0, val.strides)) and val.size > 0:
# #     raise NotImplementedError("zero strides")
#     # zero_stride_axes, = np.where(np.equal(0, val.strides))
#     # other_axes, = np.where(np.not_equal(0, val.strides))
#     # collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None) # type: ignore
#     #                           for ax in range(val.ndim))]  # type: ignore
#     # if canonicalize_types:
#     #   collapsed_val = np.asarray(
#     #       collapsed_val, dtypes.canonicalize_dtype(collapsed_val.dtype))
#     # out = mhlo.BroadcastInDimOp(
#     #     ir.RankedTensorType.get(
#     #         val.shape, dtype_to_ir_type(collapsed_val.dtype)),
#     #     _numpy_array_constant(collapsed_val, canonicalize_types=False)[0],
#     #     dense_int_elements(other_axes)).result
#     # return (out,)
#   else:
#     return _sparse_spike_vector_constant(val, canonicalize_types)


# necessary for constant type handler (e.g. see zeros above)
# TODO works for now... in the future more sophisticated implementation might be necessary (see code above)
def _sparse_spike_vector_constant_handler(val: SparseSpikeVector, canonicalize_types
                             ) -> Sequence[ir.Value]:
    return jax._src.interpreters.mlir.get_constant_handler(type(val.comb_spike_data))(val.comb_spike_data, canonicalize_types)

mlir.register_constant_handler(SparseSpikeVector, _sparse_spike_vector_constant_handler)



from jax._src.typing import ArrayLike
from jax.interpreters import pxla, xla
# from jax._src.sharding import (
#     Sharding, SingleDeviceSharding, XLACompatibleSharding, PmapSharding,
#     device_replica_id_map)
from jax._src.array import ArrayImpl, _array_shard_arg, _array_global_result_handler, _array_local_result_handler


# def _array_pmap_shard_arg(x, devices, indices, mode):
#   if dispatch.is_single_device_sharding(x.sharding):
#     return pxla._shard_device_array(x, devices, indices, mode)

#   # If the sharding of Array does not match pmap's sharding then take the slow
#   # path which is similar to what SDA does. This slow path reroute only happens
#   # for `pmap`.
#   x_indices = tuple(x.sharding.addressable_devices_indices_map(x.shape).values())
#   if indices == x_indices:
#     return [buf if buf.device() == d else buf.copy_to_device(d)
#             for buf, d in safe_zip(x._arrays, devices)]
#   else:
#     return pxla._shard_sharded_device_array_slow_path(x, devices, indices, mode)


# def _array_rest_shard_arg(x: ArrayImpl, devices, indices, mode):
#   x_indices = x.sharding.addressable_devices_indices_map(x.shape).values()
#   if not x.is_fully_addressable:
#     if tuple(x_indices) == tuple(indices):
#       return x._arrays
#     else:
#       return NotImplementedError("Cannot reshard an input that is not fully "
#                                  "addressable")
#   else:
#     if tuple(x_indices) == tuple(indices):
#       return [buf if buf.device() == d else buf.copy_to_device(d)
#               for buf, d in safe_zip(x._arrays, devices)]
#     # Resharding starts here:
#     if isinstance(x.sharding, PmapSharding):
#       return pxla.device_put(x._value, devices, replicate=True)
#     if dispatch.is_single_device_sharding(x.sharding):
#       return pxla._shard_device_array(x, devices, indices, mode)
#     else:
#       return pxla._shard_sharded_device_array_slow_path(x, devices, indices, mode)


# def _array_shard_arg(x, devices, indices, mode):
#   x._check_if_deleted()
#   if mode == pxla.InputsHandlerMode.pmap:
#     return _array_pmap_shard_arg(x, devices, indices, mode)
#   else:
#     return _array_rest_shard_arg(x, devices, indices, mode)

def _sparse_spike_vector_shard_arg(x, devices, indices, mode):
    if len(devices) > 1:
        raise NotImplementedError("Sharding `SparseSpikeVector` to multiple devices is not supported yet.")
    
    # print("_sparse_spike_vector_shard_arg")
    # print("x", x)
    # print("devices", devices)
    # print("indices", indices)
    # print("mode", mode)
    # sys.exit()
    # TODO just a quick fix, might break stuff in case of sharding across last dim
    # comb_spike_data_shard = _array_shard_arg(jax.numpy.asarray(x.comb_spike_data), devices, indices, mode)
    # comb_spike_data_shard = _array_shard_arg(x.comb_spike_data, devices, indices, mode)
    comb_spike_data_shard = pxla.shard_arg_handlers[type(x.comb_spike_data)](x.comb_spike_data, devices, indices, mode)
    # # print(comb_spike_data_shard)
    # if len(comb_spike_data_shard) > 1:
    #     raise NotImplementedError

    # out = SparseSpikeVector(comb_spike_data=comb_spike_data_shard[0], aval=x.aval)
    # # print("\n########## DONE _sparse_spike_vector_shard_arg ############")
    # # print(comb_spike_data_shard[0].flatten()[:20])
    # # print(comb_spike_data_shard[0].aval)
    # # print()
    # return [out]
    return comb_spike_data_shard

pxla.shard_arg_handlers[SparseSpikeVector] = _sparse_spike_vector_shard_arg


# def _array_global_result_handler(global_aval, out_sharding, committed,
#                                  is_out_sharding_from_xla):
#   if global_aval.dtype == dtypes.float0:
#     return lambda _: np.zeros(global_aval.shape, dtypes.float0)  # type: ignore
#   if core.is_opaque_dtype(global_aval.dtype):
#     return global_aval.dtype._rules.global_sharded_result_handler(
#         global_aval, out_sharding, committed, is_out_sharding_from_xla)
#   return lambda bufs: ArrayImpl(global_aval, out_sharding, bufs,
#                                 committed=committed, _skip_checks=True)
    

def _sparse_spike_vector_global_result_handler(global_aval, out_sharding, committed,
                                 is_out_sharding_from_xla):
    array_result_handler = _array_global_result_handler(global_aval.comb_spike_data_aval, out_sharding, committed,
                                 is_out_sharding_from_xla)
    def _global_result_handler(bufs):
        array_impl = array_result_handler(bufs)
        return SparseSpikeVector(comb_spike_data=array_impl, aval=global_aval)
    return _global_result_handler
#   if global_aval.dtype == dtypes.float0:
#     return lambda _: np.zeros(global_aval.shape, dtypes.float0)  # type: ignore
#   if core.is_opaque_dtype(global_aval.dtype):
#     return global_aval.dtype._rules.global_sharded_result_handler(
#         global_aval, out_sharding, committed, is_out_sharding_from_xla)
#   return lambda bufs: ArrayImpl(global_aval, out_sharding, bufs,
#                                 committed=committed, _skip_checks=True)
    

# pxla

pxla.global_result_handlers[AbstractSparseSpikeVector] = _sparse_spike_vector_global_result_handler
# pxla.global_result_handlers[(core.ConcreteArray, pxla.OutputType.Array)] = _array_global_result_handler
# pxla.global_result_handlers[(core.AbstractToken, pxla.OutputType.Array)] = lambda *_: lambda *_: core.token


print("pxla.global_result_handlers", pxla.global_result_handlers)


# Only used for Arrays that come out of pmap.
def _array_local_result_handler(aval, sharding, indices):
  if core.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.local_sharded_result_handler(
        aval, sharding, indices)
  return lambda bufs: ArrayImpl(aval, sharding, bufs, committed=True,
                                _skip_checks=True)

# _array_local_result_handler

def _sparse_spike_vector_local_result_handler(aval, sharding, indices):
    array_result_handler = _array_local_result_handler(aval.comb_spike_data_aval, sharding, indices)
    def _local_result_handler(bufs):
        array_impl = array_result_handler(bufs)
        return SparseSpikeVector(comb_spike_data=array_impl, aval=aval)
    return _local_result_handler

pxla.local_result_handlers[(core.ShapedArray, core.ShapedArray)] = _sparse_spike_vector_local_result_handler
# pxla.local_result_handlers[(core.ConcreteArray, pxla.OutputType.Array)] = _array_local_result_handler