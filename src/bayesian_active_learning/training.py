from functools import partial
from typing import Callable, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, random

from bayesian_active_learning.data_utils import NumpyLoader
from bayesian_active_learning.metrics import compute_model_accuracy


class Metrics(NamedTuple):
    train_accuracy_history: jnp.ndarray
    validation_accuracy_history: jnp.ndarray


# @partial(jit, static_argnums=(0, 1))
def step(
    loss: Callable[
        [optax.Params, jnp.ndarray, jnp.ndarray, random.PRNGKeyArray],
        jnp.ndarray,
    ],
    optimiser: optax.GradientTransformation,
    params: optax.Params,
    optimizer_state: optax.OptState,
    xs: jnp.ndarray,
    labels: jnp.ndarray,
    key,
) -> Tuple[optax.Params, optax.OptState]:
    grads = grad(loss)(params, xs, labels, key)
    updates, opt_state = optimiser.update(grads, optimizer_state, params)
    return optax.apply_updates(params, updates), opt_state


def fit(
    loss: Callable[
        [optax.Params, jnp.ndarray, jnp.ndarray, random.PRNGKeyArray],
        jnp.ndarray,
    ],
    params: optax.Params,
    eval_model: hk.Transformed,
    optimiser: optax.GradientTransformation,
    num_epochs: int,
    train_generator: NumpyLoader,
    validation_generator: NumpyLoader,
    key: random.PRNGKeyArray,
) -> Tuple[optax.Params, Metrics]:
    opt_state = optimiser.init(params)

    validation_accuracy_history = []
    train_accuracy_history = []

    optimiser_step = jit(partial(step, loss, optimiser))

    for _ in range(num_epochs):
        for xs, labels in train_generator:
            key, sub_key = random.split(key, 2)
            params, opt_state = optimiser_step(
                params, opt_state, xs, labels, sub_key
            )

        # compute accuracy on validation and train set
        train_accuracy = compute_model_accuracy(
            partial(eval_model.apply, params), train_generator
        )
        validation_accuracy = compute_model_accuracy(
            partial(eval_model.apply, params), validation_generator
        )

        train_accuracy_history.append(train_accuracy)
        validation_accuracy_history.append(validation_accuracy)

    return (
        params,
        Metrics(
            jnp.array(train_accuracy_history),
            jnp.array(validation_accuracy_history),
        ),
    )
