from typing import Optional, Sequence
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jrand

from jax.tree_util import tree_leaves
from jax.nn.initializers import zeros, orthogonal

from chex import Array, PRNGKey

from util import calc_mean_activity_single_val

def max_iters_not_reached(idx: int, max_iters: int) -> bool:
    if max_iters is not None:
        return idx < max_iters
    else:
        return True

def lsuv(call_fn,
        init_states,
        layerwise_model_params,
        add_call__params, 
        layer_ids_for_lsuv,
        data: Sequence[Array], 
        key: PRNGKey, 
        tgt_mean: float=-.85, 
        tgt_var: float=1., 
        mean_tol: float=.1, 
        var_tol: float=.1, 
        max_iters: Optional[int] = None):
    """
    LSUV = Layer Sequential Unit Variance
    Initialization inspired from Mishkin D and Matas J. 
    All you need is a good init. arXiv:1511.06422 [cs], February 2016.
    This is a datadriven initialization scheme for possibly stateful 
    networks which adjusts the mean and variance of the membrane output to the 
    given values for each layer. The weights and biases are required to be 
    pytree leaves with attributes `weight` and `bias`.
    It operates on an entire minibatch.
    """
    # TODO maybe write JIT compiled version?


    # weight_init_fn = lambda key, shape, dtype: jax.jit(orthogonal(), backend='cpu')(key, shape)
    w_key, b_key, key = jrand.split(key, 3)

    # Initialize all layers with `weight` or `bias` attribute with 
    # random orthogonal matrices or zeros respectively 
    layerwise_model_params = [
        [
            # weight_init_fn(w_key, weight.shape, weight.dtype),
            # jax.device_put(jax.jit(orthogonal(), backend='cpu', static_argnums=1)(w_key, weight.shape), "gpu"),
            orthogonal()(w_key, weight.shape),
            zeros(b_key, bias.shape, bias.dtype) if bias is not None else None
        ]
        for weight, bias in layerwise_model_params
    ]

    # adjust_var = lambda u_var, weight: weight *jnp.sqrt(tgt_var) / jnp.sqrt(jnp.maximum(u_var, 1e-2))
    adjust_var = lambda u_var, weight: weight * jnp.sqrt(tgt_var) / jnp.sqrt(jnp.maximum(u_var, 1e-2))
    # adjust_mean_bias = lambda u_mean, bias: bias - 0.2 * (u_mean-tgt_mean) 
    adjust_mean_bias = lambda u_mean, bias: bias - 0.1 * (u_mean-tgt_mean) 
    # (1. - .2*(mem_pot_mean - tgt_mean)/norm) # TODO Further investigation!!!
    adjust_mean_weight = lambda eps, weight: weight*(1.-eps) 

    alldone = False

    iters = 0
    while not alldone and max_iters_not_reached(iters, max_iters):
        alldone = True
        iters += 1

        states, outs = call_fn(layerwise_model_params, *add_call__params, init_states, data)
        for ilayer in layer_ids_for_lsuv:
            weights, bias = layerwise_model_params[ilayer]
            
            # Sum of spikes over entire time horizon
            mean_activity = calc_mean_activity_single_val(outs[ilayer])
            mean_activity = mean_activity[0] if isinstance(mean_activity, tuple) else mean_activity
            mem_pot_var = jnp.var(states[ilayer].U)
            mem_pot_mean = jnp.mean(states[ilayer].U)
                        
            assert mean_activity>=0
            
            print(f"Iter: {iters:4}; Layer: {ilayer}, Variance of U: {mem_pot_var:.3}, \
                    Mean of U: {mem_pot_mean:.3}, \
                    Mean Activity: {mean_activity:.3}")
            
            if jnp.isnan(mem_pot_var) or jnp.isnan(mem_pot_mean):
                done = False
                raise ValueError("NaN encountered during init!")
        
            if jnp.abs(mem_pot_var-tgt_var) > var_tol:
                layerwise_model_params[ilayer][0] = adjust_var(mem_pot_var, weights)
                done = False
            else:
                done = True
            alldone *= done
            
            if jnp.abs(mem_pot_mean-tgt_mean) > mean_tol:
                # Initialization with or without bias
                print("bias", bias is not None)
                if bias is not None:
                    layerwise_model_params[ilayer][1] = adjust_mean_bias(mem_pot_mean, bias)
                else:
                    eps = -.05*jnp.sign(mem_pot_mean-tgt_mean)/jnp.abs(mem_pot_mean-tgt_mean)**2
                    layerwise_model_params[ilayer][0] = adjust_mean_weight(eps, weights)

                done = False
            else:
                done = True
            alldone *= done
        iters += 1
            
    return [tuple(layer_params) for layer_params in layerwise_model_params]

