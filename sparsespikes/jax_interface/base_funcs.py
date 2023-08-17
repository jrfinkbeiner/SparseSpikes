import os
import sparsespikes
from sparsespikes import ssax_so_path
from .xla_gen_spike_vector import get_gen_spike_vector_fn
from .xla_spike_vector_matmul import get_spike_vector_matmul_fn

gen_spike_vector = get_gen_spike_vector_fn(
    op_name='gen_spike_vector',
    so_file=ssax_so_path,
    fn_name='gen_spike_vector_gpu_f32',
    platform='gpu',
)

spike_vector_matmul = get_spike_vector_matmul_fn(
    op_name='spike_vector_matmul',
    so_file=ssax_so_path,
    fn_name='spike_vector_matmul_gpu_f32',
    platform='gpu',
)