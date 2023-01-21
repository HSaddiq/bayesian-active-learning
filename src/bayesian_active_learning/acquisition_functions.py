from typing import Callable

import jax.numpy as jnp
from jax import jit, nn, random, vmap


def generate_logit_samples(
    model: Callable[[random.PRNGKeyArray, jnp.ndarray], jnp.ndarray],
    xs: jnp.ndarray,
    num_samples: int,
    key: random.PRNGKeyArray,
) -> jnp.ndarray:
    keys = random.split(key, num_samples)

    return vmap(model, in_axes=(0, None))(keys, xs).transpose((1, 0, 2))


def entropy(dist: jnp.ndarray) -> jnp.ndarray:
    # expect batch * num_classes
    return -jnp.sum(dist * jnp.log(dist), axis=-1)


@jit
def BALD(logit_samples: jnp.ndarray) -> jnp.ndarray:
    # expect batch * num_samples * num_classes
    probs = nn.softmax(logit_samples, axis=-1)

    posterior_predictive = jnp.mean(probs, axis=1)

    return entropy(posterior_predictive) - jnp.mean(entropy(probs), axis=1)


@jit
def max_entropy(logit_samples: jnp.ndarray) -> jnp.ndarray:
    # expect batch * num_samples * num_classes
    probs = nn.softmax(logit_samples, axis=-1)

    posterior_predictive = jnp.mean(probs, axis=1)

    return entropy(posterior_predictive)


@jit
def uniform(logit_samples: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones(logit_samples.shape[0])
