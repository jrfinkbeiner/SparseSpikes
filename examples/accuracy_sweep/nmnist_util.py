import functools as ft
from typing import Optional, Callable
import bisect
import numpy as np
# import tensorflow as tf # TODO uneccessary dependency

from multiprocessing import Pool, cpu_count
from multi_proc_helper import set_global_dataset, get_dataset_item

import tonic
import tonic.transforms as transforms
# from tfneuromorphic import nmnist

def find_first(a, tgt):
    '''
    returns the first element of tgt that is larger than a
    '''
    return bisect.bisect_left(a, tgt)


def get_tmad_slice(times, addrs, start_time, seq_len, ds_tm=1, ds_ad=1):
    '''
    Slices dataset to seq_len, return timestamp -- address array (e.g. tm, ad0, ad1, ad2 ...)
    '''
    try:
        idx_beg = find_first(times, start_time)
        idx_end = find_first(times[idx_beg:], start_time+seq_len)+idx_beg
        return np.column_stack([times[idx_beg:idx_end]//ds_tm, addrs[idx_beg:idx_end]//ds_ad])
    except IndexError:
        raise IndexError("Empty batch found")

def events_to_sparse_tensors(events,
                     order = ("x", "y", "p"),
                     deltat=1000,
                     seq_len=500,
                     sparse_size=128,
                     reduce_to_unique_spikes=False,
                     random_start=False,
                     dims=None):

    if reduce_to_unique_spikes:
        assert dims is not None
        dims_larger_one = []
        dims_larger_one_ids = []
        for i,dim in enumerate(dims):
            if dim > 1:
                dims_larger_one.append(dim)
                dims_larger_one_ids.append(i)
        ds = np.array([1] + [np.prod(dims[:i]) for i in range(1,len(dims_larger_one))], dtype=np.int16)
        def flatten_spike_ids(spike_ids):
            '''
            Flattens N-dimensional data to a 1 dimensional integer, where N = len(dims).
            dims: tuple whose elements specify the size of each dimension
            '''
            spike_ids_flat = np.sum([spike_ids[...,i]*d for i,d in zip(dims_larger_one_ids, ds)],axis=0)
            return spike_ids_flat

    times = events["t"]
    # print("\nevents_to_sparse_tensors")
    # print(f"{times.min():5}, {times.max():12}")
    if "y" in events.dtype.names:
        # addrs = np.stack([events[name], events["x"], events["y"] for name in events.dtype.names], axis=1) # TODO which order ?
        addrs = np.stack([events[name] for name in order], axis=1) # TODO which order ?
    else:
        # addrs = np.stack((events["p"], events["x"], np.zeros_like(events["x"])), axis=1) # TODO which order ?
        addrs = np.stack([np.zeros_like(events["x"]) if (name=="y") else events[name] for name in order], axis=1) # TODO which order ?
    # addrs = events[:, ["x", "y", "p"]]

    n_dims = addrs.shape[1]
    longer_than_seq_len = (times[-1] - times[0]) > (seq_len * deltat)
    t_start = times[0] if ((not random_start) or (not longer_than_seq_len)) else np.random.randint(times[0], times[-1]-seq_len*deltat)
    ts = range(t_start+deltat, t_start + (seq_len+1) * deltat, deltat)
    data = np.zeros([seq_len, sparse_size, n_dims], dtype=np.uint32)
    idx_start = 0
    idx_end = 0
    diff=0
    num_events  = np.zeros([seq_len], dtype=np.uint32)
    for i, t in enumerate(ts):
        idx_end += find_first(times[idx_end:], t)
        if idx_end > idx_start:
            if reduce_to_unique_spikes:
                ee = addrs[idx_start:idx_end]
                flat = flatten_spike_ids(ee)
                uniques, inds = np.unique(flat, return_index=True, axis=-1)
                ee = ee[inds]
            else:
                ee = addrs[idx_start:idx_end]

            #pol, x, y = ee[:, 0], (ee[:, 1] // ds_w).astype('int16'), (ee[:, 2] // ds_h).astype('int16')

            l = len(ee)
            if l>sparse_size:
                diff += len(ee)-sparse_size
                choose = np.arange(l)
                np.random.shuffle(choose)

                choose = choose[:sparse_size]
                data[i,:sparse_size,:] = ee[choose,:]
                num_events[i] = sparse_size
            else:
                data[i,:l] = ee
                num_events[i] = l

        idx_start = idx_end
    return data, num_events

# def generate_tonic_nmnist_dataset():
#     import tonic
#     import tonic.transforms as transforms

#     sensor_size = tonic.datasets.NMNIST.sensor_size

#     transform_train = transforms.Compose([
#         # transforms.Crop(target_size=(28,28)),
#         # transforms.Denoise(filter_time=10000),
#         # transforms.TimeJitter(std=10),
#         # transforms.SpatialJitter(
#         #     variance_x=0.3, # TODO originally 2
#         #     variance_y=0.3, # TODO originally 2
#         #     clip_outliers=True
#         # ),
#         transforms.ToFrame(sensor_size, time_window=1000.0),
#         # transforms.ToFrame(n_time_bins=1000),
#     ])
#     transform_test = transforms.Compose([
#         # transforms.Denoise(filter_time=10000),
#         transforms.ToFrame(sensor_size, time_window=1000.0),
#     ])

#     dataset_train = tonic.datasets.NMNIST(save_to='/Data/pgi-15/datasets',
#                                     train=True,
#                                     # transform=transform_train,
#                                     first_saccade_only=True)
#     dataset_test = tonic.datasets.NMNIST(save_to='/Data/pgi-15/datasets',
#                                     train=False,
#                                     transform=transform_test,
#                                     first_saccade_only=True)
#     return dataset_train, dataset_test


from dataclasses import dataclass
@dataclass(frozen=True)
class TimeSlice:
    seq_len: int
    randomize_start: bool = False
    def __call__(self, sequence):
        # # # print(sequence.shape)
        # slice = np.zeros((self.seq_len, *sequence.shape[1:]))
        # seq_to_use = min(self.seq_len, sequence.shape[0])
        # slice[:seq_to_use] = sequence[:seq_to_use]
        # # slice[:seq_to_use] = sequence[50:50+seq_to_use]
        # seq_to_use = min(self.seq_len, sequence.shape[0])
        # slice = sequence[:seq_to_use]

        if self.randomize_start and (sequence.shape[0]>self.seq_len):
            start = np.random.randint(0, sequence.shape[0]-self.seq_len)
        else:
            start = 0
        slice = sequence[start:start+self.seq_len]
        return slice



def get_nmnist_transforms(sparse, seq_len=300, sparse_size=None, apply_flatten=False, delta_t=1000, use_aug=False):
    sensor_size = (32, 32, 2)

    if use_aug:
        transforms_list = [transforms.RandomCrop(tonic.datasets.NMNIST.sensor_size, (32, 32))]
    else:
        transforms_list = [transforms.CenterCrop(tonic.datasets.NMNIST.sensor_size, (32, 32))]

    transforms_list.extend([
        transforms.Denoise(filter_time=10000),
        # transforms.Downsample(time_factor=1.0, spatial_factor=spatial_fac),
    ])
    if use_aug:
        transforms_list.extend([
            transforms.SpatialJitter(sensor_size=sensor_size,
                var_x=0.3,
                var_y=0.3,
                sigma_xy=0,
                clip_outliers=True),
            # transforms.TimeJitter(std=1000, clip_negative=True, sort_timestamps=True), # v1: active
            transforms.TimeSkew(coefficient=(0.9, 1.1))
        ])

    if sparse:
        transforms_list.append(
            ft.partial(events_to_sparse_tensors, deltat=delta_t,
                            seq_len=seq_len,
                            sparse_size=sparse_size,
                            dims = sensor_size,
                            reduce_to_unique_spikes = True,
                            random_start = use_aug)
        )
        if apply_flatten:
            def flatten_fn(dims, data):
                return flatten_spike_ids(dims, data[0]), data[1]
            transforms_list.append(ft.partial(flatten_fn, sensor_size))

        feature_dim = sparse_size
    else:
        transforms_list.extend([
            transforms.ToFrame(sensor_size, time_window=delta_t),
            TimeSlice(seq_len, use_aug),
            lambda x: np.clip(x, 0, 1),
            lambda x: x.reshape(x.shape[0], -1) #.astype(np.float32),
            # # transforms.ToFrame(sensor_size, n_time_bins=seq_len),
        ])
        feature_dim = np.prod(sensor_size)
    transform_train = transforms.Compose(transforms_list)    
    return transform_train, feature_dim

def create_nmnist_dataset(root, sparse, seq_len=300, sparse_size=None, dataset='train', apply_flatten=False, delta_t=1000, use_aug=False):
    '''
    root: root directory of tonic datasets
    seq_len: maximum sequence length
    dataset: 'train', 'val', or 'test
    
    returns a `tonic.datasets.NMNIST` instance
    '''
    assert dataset in ['train','val','test']
    
    if sparse:
        assert sparse_size is not None, "For `sparse=True`, `sparse_size` must be given, got `None`."

    if dataset == 'val':
        raise NotImplementedError()


    transforms, feature_dim = get_nmnist_transforms(sparse, seq_len, sparse_size, apply_flatten, delta_t, use_aug)

    dataset = tonic.datasets.NMNIST(save_to=root,
                                train=dataset == 'train',
                                transform=transforms,
                                first_saccade_only=False) # TODO decide for first saccade... has to match sparse implementation...
    return dataset, feature_dim


from typing import Tuple, Union
@dataclass
class Crop:
    """Crops events at the center to a specific output size. If output size is smaller than input
    sensor size along any dimension, padding will be used, which doesn't influence the number of
    events on that axis but just their spatial location after cropping. Make sure to use the
    cropped sensor size for any transform after CenterCrop.

    Parameters:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    sizes: Tuple[int, int]
    offsets: Tuple[int, int]

    def __init__(self, sizes: Union[int, Tuple[int, int]], offsets: Union[int, Tuple[int, int]]):
        if type(sizes) == int:
            sizes = (sizes, sizes)
        if type(offsets) == int:
            offsets = (offsets, offsets)

        self.sizes = sizes
        self.offsets = offsets

    def __call__(self, events: np.ndarray) -> np.ndarray:

        offset_idx = [max(offset, 0) for offset in self.offsets]
        cropped_events = events[
            (offset_idx[0] <= events["x"])
            & (events["x"] < (offset_idx[0] + self.sizes[0]))
            & (offset_idx[1] <= events["y"])
            & (events["y"] < (offset_idx[1] + self.sizes[1]))
        ]
        cropped_events["x"] -= self.offsets[0]
        cropped_events["y"] -= self.offsets[1]
        return cropped_events


def create_dvsgesture_dataset(root, sparse, seq_len=300, sparse_size=None, dataset='train', apply_flatten=False, delta_t=1000, use_aug=False, use_crop=False):
    '''
    root: root directory of tonic datasets
    seq_len: maximum sequence length
    dataset: 'train', 'val', or 'test
    
    returns a `tonic.datasets.NMNIST` instance
    '''
    assert dataset in ['train','val','test']
    
    if sparse:
        assert sparse_size is not None, "For `sparse=True`, `sparse_size` must be given, got `None`."

    if dataset == 'val':
        raise NotImplementedError()

    ADD_AUG = (dataset == 'train') and (use_aug == True)

    
    # scale_fac = np.array([spatial_fac, spatial_fac, 1])
    # sensor_size = tuple((np.asarray(tonic.datasets.DVSGesture.sensor_size) * scale_fac).astype(np.int16).tolist())
    sensor_size = (48, 48, 2)

    # sensor_size = (96, 96, 2)
    assert use_crop, "use_crop must be True for DVSGesture"

    if not ADD_AUG:
        transforms_list = [Crop((96, 96), offsets=(16, 32))] if use_crop else []
        spatial_fac = 0.5 if use_crop else 0.375
    else:
        transforms_list = [
            Crop((96+24, 96+24), offsets=(16-12, 32-24)),
            transforms.RandomCrop(tonic.datasets.DVSGesture.sensor_size, (96, 96))
        ]
        # transforms_list = [Crop((96, 96), offsets=(16, 32))] if use_crop else []
        spatial_fac = 0.5

    transforms_list.extend([
        transforms.Denoise(filter_time=10000),
    ])
    # if ADD_AUG:
    #     transforms_list.extend([
    #         transforms.SpatialJitter(sensor_size=sensor_size,
    #             var_x=0.6, # v1: 1.0, v2: 0.3 after downsample, v3: 1.0
    #             var_y=0.6, # v1: 1.0, v2: 0.3 after downsample, v3: 1.0
    #             sigma_xy=0,
    #             clip_outliers=True),
    #     ])
    transforms_list.extend([
        transforms.Downsample(time_factor=1.0, spatial_factor=spatial_fac),
    ])

    if ADD_AUG:
        transforms_list.extend([
            # transforms.SpatialJitter(sensor_size=sensor_size,
            #     var_x=0.3, # v1: 1.0
            #     var_y=0.3, # v1: 1.0
            #     sigma_xy=0,
            #     clip_outliers=True),
            # transforms.TimeJitter(std=1000, clip_negative=True, sort_timestamps=True), # v1: active
            transforms.TimeSkew(coefficient=(0.9, 1.1)) # v2
            # transforms.TimeSkew(coefficient=(0.8, 1.2)) # v3
        ])

    if sparse:
        transforms_list.extend([
            ft.partial(events_to_sparse_tensors, deltat=delta_t,
                            seq_len=seq_len,
                            sparse_size=sparse_size,
                            dims = sensor_size,
                            reduce_to_unique_spikes = True,
                            random_start = ADD_AUG),
        ])
        if apply_flatten:
            def flatten_fn(dims, data):
                return flatten_spike_ids(dims, data[0]), data[1]
            transforms_list.append(ft.partial(flatten_fn, sensor_size))
        feature_dim = sparse_size
    else:
        transforms_list.extend([
            transforms.ToFrame(sensor_size, time_window=delta_t),
            TimeSlice(seq_len, ADD_AUG),
            lambda x: np.clip(x, 0, 1),
            lambda x: x.reshape(x.shape[0], -1) # .astype(np.float32),
        ])
        # transform_test = transforms.Compose([
        #     # transforms.Denoise(filter_time=10000),
        #     # transforms.ToFrame(sensor_size, time_window=1000.0),
        #     transforms.ToFrame(sensor_size, n_time_bins=seq_len),
        # ])
        feature_dim = np.prod(sensor_size)
    transform_train = transforms.Compose(transforms_list)

    dataset = tonic.datasets.DVSGesture(save_to=root,
                                train=dataset == 'train',
                                transform=transform_train) # TODO decide for first saccade... has to match sparse implementation...
    return dataset, feature_dim


def create_shd_dataset(root, sparse, seq_len=1000, sparse_size=None, dataset='train', apply_flatten=False, delta_t=1000, use_aug=False):
    '''
    root: root directory of tonic datasets
    seq_len: maximum sequence length
    dataset: 'train', 'val', or 'test
    
    returns a `tonic.datasets.SHD` instance
    '''
    assert dataset in ['train','val','test']
    
    if sparse:
        assert sparse_size is not None, "For `sparse=True`, `sparse_size` must be given, got `None`."

    if dataset == 'val':
        raise NotImplementedError()
    
    sensor_size = tonic.datasets.SHD.sensor_size


    transforms_list = [
        # transforms.Denoise(filter_time=10000)
    ]

    ADD_AUG = (dataset == 'train') and (use_aug == True)
    if ADD_AUG:
        transforms_list.extend([
            # transforms.SpatialJitter(sensor_size=sensor_size,
            #     var_x=0.3, # v1: 1.0
            #     var_y=0.3, # v1: 1.0
            #     sigma_xy=0,
            #     clip_outliers=True),
            transforms.TimeJitter(std=20, clip_negative=False, sort_timestamps=True), # v1: active
            transforms.TimeSkew(coefficient=(0.8, 1.2))
        ])

    if sparse:
        transforms_list.extend([
            # transforms.Denoise(filter_time=10000),
            ft.partial(events_to_sparse_tensors, deltat=delta_t,
                            seq_len=seq_len,
                            sparse_size=sparse_size,
                            dims = sensor_size,
                            reduce_to_unique_spikes = True,
                            random_start = ADD_AUG),
        ])
        if apply_flatten:
            def flatten_fn(dims, data):
                return flatten_spike_ids(dims, data[0]), data[1]
            transforms_list.append(ft.partial(flatten_fn, sensor_size))
        feature_dim = sparse_size
    else:
        transforms_list.extend([
            transforms.ToFrame(sensor_size, time_window=delta_t),
            TimeSlice(seq_len, ADD_AUG),
            lambda x: np.clip(x, 0, 1),
            # transforms.ToFrame(sensor_size, n_time_bins=seq_len),
            lambda x: x.reshape(x.shape[0], -1)
        ])
        feature_dim = np.prod(sensor_size)
    transform_train = transforms.Compose(transforms_list)
    dataset = tonic.datasets.SHD(save_to=root,
                                train=dataset == 'train',
                                transform=transform_train) # TODO decide for first saccade... has to match sparse implementation...
    return dataset, feature_dim



def create_dense_batch(ids, dataset, seq_len, batch_size, feature_dim, pool=None):
    assert len(ids) == batch_size
    batched_data = np.zeros((batch_size, seq_len, feature_dim), dtype=np.float32)
    batched_labels = np.empty(batch_size, dtype=np.int32)
    if pool is None:
        for i,idx in enumerate(ids):
            datai, label = dataset[idx]
            batched_data[i,:min(seq_len, datai.shape[0]),:] = datai[:seq_len,:]
            batched_labels[i] = label
    else:
        data = pool.map(get_dataset_item, ids)
        for i,(datai,label) in enumerate(data):
            batched_data[i,:min(seq_len, datai.shape[0]),:] = datai[:seq_len,:]
            batched_labels[i] = label
    return {"inp_spikes": batched_data, "targets": batched_labels}        

def create_sparse_batch(ids, dataset, seq_len, batch_size, feature_dim, pool=None): #, seq_len):
    assert len(ids) == batch_size
    batched_spike_ids = np.zeros((batch_size, seq_len, feature_dim), dtype=np.float32)
    batched_spike_num_spikes = np.zeros((batch_size, seq_len), dtype=np.float32)
    batched_labels = np.empty(batch_size, dtype=np.int32)
    if pool is None:
        for i,idx in enumerate(ids):
            data, label = dataset[idx]
            batched_spike_ids[i,:min(seq_len, data[0].shape[0]),:] = data[0][:seq_len,:]
            batched_spike_num_spikes[i,:min(seq_len, data[1].shape[0])] = data[1][:seq_len]
            batched_labels[i] = label
    else:
        data = pool.map(get_dataset_item, ids)
        for i,((inp_spike_ids,num_inp_spikes),label) in enumerate(data):
            batched_spike_ids[i,:min(seq_len, inp_spike_ids.shape[0]),:] = inp_spike_ids[:seq_len,:]
            batched_spike_num_spikes[i,:min(seq_len, num_inp_spikes.shape[0])] = num_inp_spikes[:seq_len]
            batched_labels[i] = label
    return {"inp_spike_ids": batched_spike_ids, "num_inp_spikes": batched_spike_num_spikes, "targets": batched_labels}


# def create_nmnist_gener(root, sparse, num_epochs=1, seq_len=300, sparse_size=None, num_samples=None, dataset='train', shuffle=None, batchsize=None, use_multiprocessing=False):
#     '''
#     root: root directory of tonic datasets
#     seq_len: maximum sequence length
#     dataset: 'train', 'val', or 'test
    
#     returns a generator function with yields data, num_events, target
#     target: integer
#     data: flattened float32 array of dimension seq_len x prod(sersor_size) containing flattened event addresses
#     '''

#     return create_gener("NMNIST", root, sparse, num_epochs=num_epochs, seq_len=seq_len, sparse_size=sparse_size, num_samples=num_samples, dataset_split=dataset, shuffle=shuffle, batchsize=batchsize, use_multiprocessing=use_multiprocessing)

def get_create_dataset_fn(dataset_name, **kwargs):
    dataset_to_fn = {
        "NMNIST": create_nmnist_dataset,
        "SHD": create_shd_dataset,
        "DVSGesture": ft.partial(create_dvsgesture_dataset, **kwargs),
    }
    return dataset_to_fn[dataset_name]


def create_gener(rng: np.random.Generator, dataset_name, root, sparse, seq_len=300, sparse_size=None, num_samples=None, dataset_split='train', shuffle=None, batchsize=None, use_multiprocessing=False, delta_t=1000, use_aug=False, use_crop=False):
    '''
    root: root directory of tonic datasets
    seq_len: maximum sequence length
    dataset: 'train', 'val', or 'test
    
    returns a generator function with yields data, num_events, target
    target: integer
    data: flattened float32 array of dimension seq_len x prod(sersor_size) containing flattened event addresses
    '''

    dataset, feature_dim = get_create_dataset_fn(dataset_name, use_crop=use_crop)(root, sparse, seq_len=seq_len, sparse_size=sparse_size, dataset=dataset_split, apply_flatten=True, delta_t=delta_t, use_aug=use_aug)

    if shuffle is None:
        shuffle = True if dataset_split == 'train' else False

    if num_samples is None:
        num_samples = len(dataset)
    if batchsize is not None:
        num_batches = int((num_samples//batchsize))

    if use_multiprocessing:
        assert batchsize is not None

    if shuffle:
        idx_samples = rng.choice(len(dataset), num_samples, replace=False)
    else:
        idx_samples = np.arange(num_samples)

    # idx_samples = np.empty(num_samples*num_epochs, dtype=np.int64)
    # for iepoch in range(num_epochs):
    #     if shuffle:
    #         np.random.shuffle(idx_samples_base)
    #     idx_samples[iepoch*num_samples:(iepoch+1)*num_samples] = idx_samples_base

    def shuffle_warpper(generator_fn):
        def gen():
            if shuffle: 
                rng.shuffle(idx_samples)
            for data in generator_fn():
                yield data
        return gen

    def gen_dense_batched():
        for ibatch in range(num_batches):
            inds = idx_samples[ibatch*batchsize:(ibatch+1)*batchsize]
            ret_data = create_dense_batch(inds, dataset, seq_len, batchsize, feature_dim)
            yield ret_data

    def gen_dense_batched_multiproc():
        with Pool(min(cpu_count(), 8), initializer=set_global_dataset, initargs=(dataset,)) as p:
            for ibatch in range(num_batches):
                inds = idx_samples[ibatch*batchsize:(ibatch+1)*batchsize]
                ret_data = create_dense_batch(inds, dataset, seq_len, batchsize, feature_dim, p)
                yield ret_data

    def gen_dense():
        for i in idx_samples:
            data, label = dataset[i]
            yield {"inp_spikes": data, "targets": label}

    def gen_sparse_batched():
        for ibatch in range(num_batches):
            inds = idx_samples[ibatch*batchsize:(ibatch+1)*batchsize]
            ret_data = create_sparse_batch(inds, dataset, seq_len, batchsize, feature_dim)
            yield ret_data

    def gen_sparse_batched_multiproc():
        # use at most one process per cpu or one process per sample
        with Pool(min(cpu_count(), 8), initializer=set_global_dataset, initargs=(dataset,)) as p:
            for ibatch in range(num_batches):
                inds = idx_samples[ibatch*batchsize:(ibatch+1)*batchsize]
                ret_data = create_sparse_batch(inds, dataset, seq_len, batchsize, feature_dim, p)
                yield ret_data

    def gen_sparse():    
        for i in idx_samples:
            data, label = dataset[i]
            yield {"inp_spike_ids": data[0].astype(np.uint32), "num_inp_spikes": data[1].astype(np.uint32), "targets": label}

    if batchsize is None:
        gen = gen_sparse if sparse else gen_dense 
    else:
        if use_multiprocessing:
            # TODO implement multiprocessing !!!
            # gen = gen_sparse_batched_multiproc if sparse else gen_dense_batched_multiproc
            gen = gen_sparse_batched_multiproc if sparse else gen_dense_batched_multiproc
        else:
            gen = gen_sparse_batched if sparse else gen_dense_batched

    return shuffle_warpper(gen), num_samples
    # return gen, num_samples


# class KerasNMNIST(tf.keras.utils.Sequence):
#     _sparse: bool = False
#     _gen_batch: Callable

#     def __init__(self, 
#             dataset, 
#             batch_size: int, 
#             sparse: bool,
#             shuffle: Optional[bool] = False, 
#             rng: Optional[np.random.Generator] = None, 
#             # processing_func: Optional[Callable] = None
#         ):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         # self.processing_func = processing_func
#         self.sparse = sparse
#         self.shuffle = shuffle
#         if shuffle:
#             assert isinstance(rng, np.random.Generator), f"If `shuffle=True`, `rng` has to be an instance of `numpy.random.Generator`, got '{rng}'."
#         self.rng = rng
#         self._indices = np.arange(len(self.dataset))

#     @property
#     def sparse(self):
#         return self._sparse

#     @sparse.setter
#     def sparse(self, value):
#         self._sparse = value
#         self._gen_batch = self._gen_sparse_batch if value else self._gen_dense_batch

#     def _gen_dense_batch(self, ids):
#         return create_dense_batch(ids, self.dataset, self.batch_size)

#     def _gen_sparse_batch(self, ids):
#         return create_sparse_batch(ids, self.dataset, self.batch_size)

#     def __len__(self):
#         return int(len(self.dataset) // self.batch_size)

#     def __getitem__(self, idx):
#         inds = self._indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#         data_batch = self._gen_batch(inds)
#         return data_batch
        
#     def on_epoch_end(self):
#         if self.shuffle:
#             self.rng.shuffle(self._indices)

# def get_nmnist_keras_dataset(rng, root, sparse, batchsize, seq_len, sparse_size: Optional[int] = None):
#     if sparse:
#         assert sparse_size is not None

#     tonic_dataset = create_nmnist_dataset(root, sparse, seq_len=seq_len, sparse_size=sparse_size, dataset='train')
#     keras_dataset = KerasNMNIST(
#             dataset=tonic_dataset, 
#             batch_size=batchsize, 
#             sparse=sparse,
#             shuffle=True, 
#             rng=rng, 
#     )
#     return keras_dataset


# def get_nmnist_dataset(root, sparse, num_epochs, seq_len, inp_dim, batchsize, num_samples=None, dims=None, multiprocessing=False):
#     if multiprocessing:
#         gen_train, num_samples = create_nmnist_gener(root, sparse, num_epochs, seq_len=seq_len, sparse_size=inp_dim, num_samples=num_samples, batchsize=batchsize, multiprocessing=multiprocessing)
#     else:
#         gen_train, num_samples = create_nmnist_gener(root, sparse, num_epochs, seq_len=seq_len, sparse_size=inp_dim, num_samples=num_samples, multiprocessing=multiprocessing)
#     # get_train, num_samples = create_nmnist_gener(root, sparse, seq_len=seq_len, sparse_size=sparse_size, num_samples=num_samples)

#     # dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, inp_dim), dtype=tf.float32),
#     #                                                                         tf.TensorSpec(shape=(), dtype=tf.int32))))

#     if dims is None:
#         dims = (34,34,2)
#     flatten_fn = ft.partial(flatten_data_tf, dims=dims)
#     if multiprocessing:
#         if sparse:
#             dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spike_ids": tf.TensorSpec(shape=(batchsize, seq_len, inp_dim), dtype=tf.float32),
#                                                                                     "num_inp_spikes": tf.TensorSpec(shape=(batchsize, seq_len, ), dtype=tf.int32),
#                                                                                     "targets": tf.TensorSpec(shape=(batchsize, ), dtype=tf.int32)})
#         else:
#             dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spikes": tf.TensorSpec(shape=(batchsize, seq_len, inp_dim), dtype=tf.float32),
#                                                                                     "targets": tf.TensorSpec(shape=(batchsize, ), dtype=tf.int32)})
#     else:
#         if sparse:
#             dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spike_ids": tf.TensorSpec(shape=(seq_len, inp_dim, len(dims)), dtype=tf.float32),
#                                                                                     "num_inp_spikes": tf.TensorSpec(shape=(seq_len, ), dtype=tf.int32),
#                                                                                     "targets": tf.TensorSpec(shape=(), dtype=tf.int32)})
#         else:
#             dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spikes": tf.TensorSpec(shape=(seq_len, inp_dim), dtype=tf.float32),
#                                                                                     "targets": tf.TensorSpec(shape=(), dtype=tf.int32)})


#     # idx = np.arange(num_samples)
#     # # dataset = tf.data.Dataset.from_tensor_slices(idx)
#     # dataset = tf.data.Dataset.range(num_samples) # .as_numpy_iterator()

#     # dataset = dataset.map(lambda *args: args, num_parallel_calls=tf.data.AUTOTUNE)
#     if not multiprocessing:
#         if sparse:
#             # TODO perform transform here instead of inside tonic dataset?, makes use of `num_parallel_calls`
#             dataset = dataset.map(flatten_fn, num_parallel_calls=tf.data.AUTOTUNE)
#         dataset = dataset.batch(batchsize, drop_remainder=True)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     return dataset




def flatten_spike_ids(dims, spike_ids):
    '''
    Flattens N-dimensional data to a 1 dimensional integer, where N = len(dims).
    dims: tuple whose elements specify the size of each dimension
    '''
    dims_larger_one = []
    dims_larger_one_ids = []
    for i,dim in enumerate(dims):
        if dim > 1:
            dims_larger_one.append(dim)
            dims_larger_one_ids.append(i)
    ds = np.array([1] + [np.prod(dims[:i]) for i in range(1,len(dims_larger_one))], dtype=np.int16)
    spike_ids_flat = np.sum([spike_ids[...,i]*d for i,d in zip(dims_larger_one_ids, ds)],axis=0, dtype=spike_ids.dtype)
    return spike_ids_flat

# # def flatten_spike_ids(sensor_dim, ids):
# #     print("\nflatten_spike_ids")
# #     print(sensor_dim)
# #     print(ids)
# #     dimx, dimy, dimp = sensor_dim
# #     # return  ids[...,1] + dimy * (ids[...,2] + dimx * ids[...,0])
# #     # return  ids[...,1] + dimx * (ids[...,2] + dimy * ids[...,0])
# #     return  ids[...,0] + dimy * (ids[...,1] + dimx * ids[...,2])


# def flatten_data_tf(data, dims):
#     '''
#     Flattens N-dimensional data to a 1 dimensional integer, where N = len(dims).
#     dims: tuple whose elements specify the size of each dimension
#     '''
#     data["inp_spike_ids"] = flatten_spike_ids(dims, data["inp_spike_ids"])
#     return data

# # def cast_data_tf(input_data, label):
# #     data, num_events = input_data
# # # def cast_data_tf(data, num_events, label):
# #     return (tf.cast(data, tf.float32), tf.expand_dims(tf.cast(num_events, tf.int32), axis=-1)), tf.cast(label, tf.int32) # cast label only because of IPUModel

# # def get_nmnist_dataset(hdf5_filepath, sparse_size, num_samples, batchsize, dims=None, seq_len=300, sparse=True):

# #     assert sparse, "Currently only the sparse nmnist dataset generator is implemented"

# #     if dims is None:
# #         dims = (2,34,34)
# #     dtype = tf.int16

# #     # gen_train = nmnist.create_gener('/Data/pgi-15/datasets/nmnist/n_mnist.hdf5', dataset='train', sparse_size = sparse_size, num_samples=num_samples, seq_len=seq_len)
# #     gen_train = nmnist.create_gener(hdf5_filepath, dataset='train', sparse_size = sparse_size, num_samples=num_samples, seq_len=seq_len)
# #     # it = gen_train()
# #     # data, num_events, label = next(it)
# #     # data_flat, _, _ = nmnist.flatten_data(data, num_events, label, dims = dims)
# #     #  = nmnist.sparse_vector_to_dense(data, num_events, dims = dims)
    
# #     dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, sparse_size, len(dims)), dtype=dtype),
# #                                                                            tf.TensorSpec(shape=(seq_len), dtype=dtype)),
# #                                                                           tf.TensorSpec(shape=(), dtype=dtype)))

# #     flatten_fn = ft.partial(flatten_data_tf, dims=dims)
# #     print(dataset)

# #     # dataset = dataset.shuffle()
# #     # dataset = dataset.prefetch(16) # TODO why 16 32?
# #     # dataset = dataset.map(nmnist.flatten_data_tf)
# #     dataset = dataset.map(flatten_fn, num_parallel_calls=tf.data.AUTOTUNE)
# #     dataset = dataset.map(cast_data_tf, num_parallel_calls=tf.data.AUTOTUNE)
# #     dataset = dataset.repeat()
# #     dataset = dataset.shuffle(num_samples, reshuffl32e_each_iteration=False)
# #     dataset = dataset.batch(batchsize, drop_remainder=True)
# #     dataset = dataset.prefetch(tf.data.AUTOTUNE)
# #     return dataset


def load_dataset_to_tensor_dict(dataset_name, root, sparse, seq_len, inp_dim, num_samples=None, iter_batchsize=None, shuffle=True):

    if iter_batchsize is None:
        iter_batchsize = 1000
    gen, num_samples = create_gener(dataset_name, root, sparse, 1, seq_len=seq_len, sparse_size=inp_dim, dataset_split="train", num_samples=num_samples, batchsize=iter_batchsize, shuffle=shuffle, use_multiprocessing=True)

    assert num_samples % iter_batchsize == 0, "`num_samples` must be divisible by `iter_batchsize`"

    if sparse:

        # TODO apply flatten here or in create_nmnist_gener !!!

        inp_spike_ids = np.empty((num_samples, seq_len, inp_dim), dtype=np.float32)
        num_inp_spikes = np.empty((num_samples, seq_len, 1), dtype=np.int32)
        labels = np.empty((num_samples,), dtype=np.int32)
        for i,data in enumerate(gen()):
            inp_spike_ids[i*iter_batchsize:(i+1)*iter_batchsize] = data["inp_spike_ids"]
            num_inp_spikes[i*iter_batchsize:(i+1)*iter_batchsize] = np.expand_dims(data["num_inp_spikes"], axis=-1)
            labels[i*iter_batchsize:(i+1)*iter_batchsize] = data["targets"]
        ret_val = {
            "inp_spike_ids": inp_spike_ids,
            "num_inp_spikes": num_inp_spikes,
            "targets": labels,
        }
    else:
        inp_spikes = np.empty((num_samples, seq_len, inp_dim), dtype=np.float32)
        labels = np.empty((num_samples,), dtype=np.int32)
        for i,data in enumerate(gen()):
            inp_spikes[i*iter_batchsize:(i+1)*iter_batchsize] = data["inp_spikes"]
            labels[i*iter_batchsize:(i+1)*iter_batchsize] = data["targets"]
        ret_val = {
            "inp_spikes": inp_spikes,
            "targets": labels,
        }
    return ret_val


# import numpy as np
# from torch.utils import data
# from torchvision.datasets import MNIST

# def numpy_collate(batch):
#   if isinstance(batch[0], np.ndarray):
#     return np.stack(batch)
#   elif isinstance(batch[0], (tuple,list)):
#     transposed = zip(*batch)
#     return [numpy_collate(samples) for samples in transposed]
# #   elif isinstance(batch[0], dict):
# #     return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
#     # values, keys = zip(*batch[0].items())
#   else:
#     return np.array(batch)

# class SpikesLoader(data.DataLoader):
#   def __init__(self, dataset, batch_size=1,
#                 shuffle=False, sampler=None,
#                 batch_sampler=None, num_workers=0,
#                 pin_memory=False, drop_last=False,
#                 timeout=0, worker_init_fn=None):
#     super(self.__class__, self).__init__(dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         sampler=sampler,
#         batch_sampler=batch_sampler,
#         num_workers=num_workers,
#         # collate_fn=numpy_collate,
#         pin_memory=pin_memory,
#         drop_last=drop_last,
#         timeout=timeout,
#         worker_init_fn=worker_init_fn)

# # class FlattenAndCast(object):
# #   def __call__(self, pic):
# #     return np.ravel(np.array(pic, dtype=jnp.float32))


# def get_dataloader(rng: np.random.Generator, dataset_name, root, sparse, seq_len=300, sparse_size=None, dataset_split='train', shuffle=None, batchsize=None, delta_t=1000, **kwargs):
#     dataset = get_create_dataset_fn(dataset_name)(root, sparse, seq_len=seq_len, sparse_size=sparse_size, dataset=dataset_split, apply_flatten=True, delta_t=delta_t)
#     dataloader = SpikesLoader(
#         dataset,
#         batch_size=batchsize,
#         shuffle=shuffle,
#         **kwargs
#     )
#     return dataloader, len(dataset)


# def get_tonic_prototyping_dataloader(rng: np.random.Generator, dataset_name, root, sparse, seq_len=300, sparse_size=None, dataset_split='train', shuffle=None, batchsize=None, delta_t=1000, **kwargs):

#     assert dataset_name == "NMNIST", "Only NMNIST is supported for now"
#     from tonic.prototype.datasets.nmnist import NMNIST
#     transforms = get_nmnist_transforms(sparse, seq_len, sparse_size, True, delta_t)
#     import os
#     datapipe = NMNIST(
#         root=os.path.join(root, "NMNIST"),
#         transform=transforms,
#         target_transform=None,
#         transforms=None,
#         train=(dataset_split == "train"),
#         first_saccade_only=False,
#     ) 
#     dataloader = SpikesLoader(
#         datapipe,
#         batch_size=batchsize,
#         shuffle=shuffle,
#         **kwargs
#     )

#     # from torchdata.datapipes.iter import Mapper
#     # datapipe = NMNIST(root="./data")
#     # datapipe = Mapper(datapipe, t, input_col=0) # input_col=0 tells Mapper to apply the function t to the entry number 0 of the tuple. 
#     # frames, target = next(iter(datapipe))

#     return dataloader, len(datapipe)



if __name__ == "__main__":
    import sys
    datapath_root = ... # TODO set
    gens = {}
    data = {}
    rng = np.random.default_rng(42)
    # for use_sparse in [True, False]:
    for use_sparse in [False]:
        sparse_str = "sparse" if use_sparse else "dense"
        # gen, num_samples = create_nmnist_gener(
        gen, num_samples = create_gener(
            # "NMNIST",
            # "SHD",
            rng,
            "DVSGesture",
            root=datapath_root, 
            # root="/localdata/datasets/", 
            sparse=use_sparse, 
            # num_epochs=1, 
            seq_len=400, 
            sparse_size=128*4, 
            # num_samples=None, 
            # dataset='train', 
            dataset_split='train', 
            shuffle=False, 
            batchsize=100, 
            use_multiprocessing=True,
            delta_t=1000,
            use_aug=True,
        )
        gens[sparse_str] = gen
        data_next = next(gen())
        data[sparse_str] = data_next

        print()
        if use_sparse:
            num_inp_spikes = data["sparse"]["num_inp_spikes"]
            print("max spike id: ", data["sparse"]["inp_spike_ids"].max())
        else:
            print(data["dense"]["inp_spikes"].min(), data["dense"]["inp_spikes"].max())
            data["dense"]["inp_spikes"] = np.clip(data["dense"]["inp_spikes"], 0, 1)
            print(data["dense"]["inp_spikes"].min(), data["dense"]["inp_spikes"].max())
            num_inp_spikes = data["dense"]["inp_spikes"].sum(axis=2).astype(np.int32)
       
        print(num_inp_spikes.shape)
        print(num_inp_spikes)
        print(num_inp_spikes.mean(), num_inp_spikes.std(), num_inp_spikes.min(), num_inp_spikes.max())
        # sys.exit()

        import matplotlib.pyplot as plt

        # print(data["dense"]["inp_spikes"].shape)
        # for i in range(5): 
        #     plt.figure()
        #     plt.imshow(data["dense"]["inp_spikes"][10+i].sum(axis=0))
        # plt.show()
        # # sys.exit()

        
        plt.hist(num_inp_spikes.flatten(), bins=np.arange(0, num_inp_spikes.max()+1, 1))
        plt.show()
        sys.exit()

    print()
    print(data["dense"]["inp_spikes"].shape)
    print(data["dense"]["inp_spikes"].sum(axis=2).astype(np.int32))
    print()
    print(data["sparse"]["num_inp_spikes"].shape)
    print(data["sparse"]["num_inp_spikes"])
    print()
    print(data["dense"]["inp_spikes"].shape)
    print(data["dense"]["inp_spikes"])
    # sys.exit()

    num_inp_spikes_dense = data["dense"]["inp_spikes"].sum(axis=2).astype(np.int32)[0]
    num_inp_spikes_sparse = data["sparse"]["num_inp_spikes"][0]
    inp_spikes_ids_dense = np.argwhere(data["dense"]["inp_spikes"][0] > 0)
    inp_spikes_ids_sparse = data["sparse"]["inp_spike_ids"][0]

    print()
    print(np.all(num_inp_spikes_dense[:-1] == num_inp_spikes_sparse[1:]))
    print()
    print(inp_spikes_ids_dense.shape)
    print(inp_spikes_ids_dense[:50])
    print()
    print(inp_spikes_ids_sparse.shape)
    print(inp_spikes_ids_sparse[:8, :30])