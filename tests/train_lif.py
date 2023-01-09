from test_lif import calc_loss_batch



def update():
    Nc = 2 # Number of classes
    N = [4, 8, Nc] # List of number of neurons per layer
    T = 5 # Number of timesteps per epoch
    BATCHSIZE = 12
    SEED = 42 

    rng = np.random.default_rng(SEED)
    keys = jrandom.split(jrandom.PRNGKey(rng.integers(999999)), 2)
    weights = init_network_weights(rng, N, False)
    NUM_LAYERS = len(N)-1
    alphas, betas = [0.95]*NUM_LAYERS, [0.9]*NUM_LAYERS

    initial_state = init_network_states(BATCHSIZE, N[1:])
    target = rng.integers(0, Nc, size=(BATCHSIZE,))
    targets_one_hot = create_one_hot(target, Nc, dtype=jnp.float32)

    inp_states = jrandom.uniform(keys[0], (T, BATCHSIZE, N[0]), dtype=jnp.float32)*1.5
    thresholds = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
    inp_spikes_dense = jnp.asarray(inp_states > thresholds[0], dtype=jnp.float32)
    inp_spikes_sparse_list = [gen_spike_vector(inp_stat, thresholds, max_num_spikes=N[0]) for inp_stat in inp_states] 


def main():


if __name__ == '__main__':
    main()