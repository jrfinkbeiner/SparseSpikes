import functools as ft
import jax
import jax.numpy as jnp
import numpy as np

NUM_CLASSES = 4
out = jnp.asarray(np.random.random((10, 12, NUM_CLASSES)))
# out = jnp.asarray(np.random.randint(NUM_CLASSES, size=(10, NUM_CLASSES)))

print("out", out)

def calc_pred_from_linear_readout(out, *, num_classes):
    preds_per_timestep = jnp.argmax(out, axis=-1)
    vals, counts = jnp.unique(preds_per_timestep, return_counts=True, size=num_classes)
    return vals[counts.argmax(axis=0)]

preds = jax.vmap(ft.partial(calc_pred_from_linear_readout, num_classes=NUM_CLASSES), in_axes=(1,))(out)

print("preds", preds)

def calc_accuracy_from_linear_ro(out, target, normalized=True, *, num_classes):
    preds = jax.vmap(ft.partial(calc_pred_from_linear_readout, num_classes=num_classes), in_axes=(1,))(out)
    if normalized:
        return (preds==target).mean()
    else:
        return (preds==target).sum()