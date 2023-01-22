import haiku as hk
import jax.numpy as jnp
import optax
from jax import random


def classification_loss(
    model: hk.Transformed,
    params: optax.Params,
    xs: jnp.ndarray,
    labels: jnp.ndarray,
    key: random.PRNGKeyArray,
) -> jnp.ndarray:
    y_hat = model.apply(params, x=xs, rng=key)

    # optax also provides a number of common loss functions.
    loss_value = optax.softmax_cross_entropy(y_hat, labels)

    return loss_value.mean()
