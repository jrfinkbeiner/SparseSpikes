import jax
from sparsespikes.jax_interface import check_is_sparse_spikes_type

def calc_mean_activity_single_val(spikes):
    mean_activity = (spikes.num_spikes[..., 0, :].mean().item()/spikes.num_neurons, spikes.num_spikes[..., 1, :].mean().item()/spikes.num_neurons) if check_is_sparse_spikes_type(spikes) else spikes.mean().item()
    return mean_activity

def calc_mean_activity(spikes):
    return jax.tree_map(calc_mean_activity_single_val, spikes)