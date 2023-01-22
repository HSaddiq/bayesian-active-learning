from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jit, lax, nn, random, vmap

from bayesian_active_learning.data_utils import NumpyDataset, NumpyLoader


def acquire_new_data(
    model: Callable[[random.PRNGKeyArray, jnp.ndarray], jnp.ndarray],
    acquisition_function: Callable[[jnp.ndarray], jnp.ndarray],
    num_predictive_samples: int,
    num_acquired_points: int,
    pool_dataset: NumpyDataset,
    train_dataset: NumpyDataset,
    key: random.PRNGKeyArray,
) -> Tuple[NumpyDataset, NumpyDataset]:
    # acquire new points from the pool set to add to the train set

    current_train_X, current_train_y = train_dataset.arrays
    current_pool_X, current_pool_y = pool_dataset.arrays
    pool_dataloader = NumpyLoader(dataset=pool_dataset, batch_size=256)

    # first compute acquisition scores
    acquisition_scores = []

    for xs, _ in pool_dataloader:
        logits = generate_logit_samples(
            model, xs, num_samples=num_predictive_samples, key=key
        )
        acquisition_scores.append(acquisition_function(logits))

    all_acquisition_scores = jnp.concatenate(acquisition_scores)

    # get top "num acquired points" and select points
    _, indices = lax.top_k(all_acquisition_scores, num_acquired_points)

    acquired_X = current_pool_X[indices]
    acquired_y = current_pool_y[indices]

    # create updated train and pool datasets

    mask = jnp.ones(len(current_pool_X), jnp.bool_)
    mask = mask.at[indices].set(False)

    new_pool_X = current_pool_X[mask]
    new_pool_y = current_pool_y[mask]
    new_pool_dataset = NumpyDataset(new_pool_X, new_pool_y)

    new_train_X = jnp.concatenate([current_train_X, acquired_X])
    new_train_y = jnp.concatenate([current_train_y, acquired_y])
    new_train_dataset = NumpyDataset(new_train_X, new_train_y)

    return (new_train_dataset, new_pool_dataset)


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


def get_acquisition_function(
    name: str,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    map = {"BALD": BALD, "Random": uniform, "Max Entropy": max_entropy}

    return map[name]
