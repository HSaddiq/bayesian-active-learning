import copy
from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, random

from bayesian_active_learning.acquisition_functions import acquire_new_data
from bayesian_active_learning.data_utils import NumpyDataset, NumpyLoader
from bayesian_active_learning.losses import classification_loss
from bayesian_active_learning.metrics import compute_model_accuracy
from bayesian_active_learning.models import model
from bayesian_active_learning.training import Metrics, fit


def experiment_run(
    train_set: Tuple[jnp.ndarray, jnp.ndarray],
    val_set: Tuple[jnp.ndarray, jnp.ndarray],
    pool_set: Tuple[jnp.ndarray, jnp.ndarray],
    test_set: Tuple[jnp.ndarray, jnp.ndarray],
    weight_decay: float,
    acquisition_fn: Callable[[jnp.ndarray], jnp.ndarray],
    num_predictive_samples: int,
    num_acquired_points_per_iteration: int,
    num_iterations: int,
    key: random.PRNGKeyArray,
) -> jnp.ndarray:

    # generate datasets for each subset of data
    train_dataset = NumpyDataset(*train_set)
    pool_dataset = NumpyDataset(*pool_set)
    validation_dataset = NumpyDataset(*val_set)
    test_dataset = NumpyDataset(*test_set)

    validation_generator = NumpyLoader(
        dataset=validation_dataset, batch_size=256
    )

    test_generator = NumpyLoader(dataset=test_dataset, batch_size=256)

    # create, transform and intialise model (and evaluation model)
    num_classes = train_set[1].shape[1]
    dropout_rates = (0.25, 0.5)

    base_training_model = partial(model, num_classes, dropout_rates)
    stochastic_model = hk.transform(base_training_model)

    base_eval_model = partial(model, num_classes, (0, 0))
    eval_model = hk.without_apply_rng(hk.transform(base_eval_model))

    key, subkey = random.split(key)
    params = stochastic_model.init(subkey, jnp.zeros((1, 28, 28)))

    loss = partial(classification_loss, stochastic_model)

    # train the model using the initial training data
    optimiser = optax.adamw(1e-3, weight_decay=weight_decay)

    key, subkey = random.split(key)
    training_generator = NumpyLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
    )

    params, _ = fit(
        loss=loss,
        params=params,
        eval_model=eval_model,
        optimiser=optimiser,
        num_epochs=100,
        train_generator=training_generator,
        validation_generator=validation_generator,
        key=subkey,
    )

    pretrained_params = copy.deepcopy(params)

    # obtain accuracy for initially trained model on test dataset
    test_accuracy_history = []

    test_accuracy_history.append(
        compute_model_accuracy(
            partial(eval_model.apply, pretrained_params), test_generator
        )
    )

    # run the active learning procedure
    for iteration in range(num_iterations):
        # update train and pool dataset using acquisition function
        key, subkey = random.split(key, 2)

        train_dataset, pool_dataset = acquire_new_data(
            jit(partial(stochastic_model.apply, params)),
            acquisition_function=acquisition_fn,
            num_predictive_samples=num_predictive_samples,
            num_acquired_points=num_acquired_points_per_iteration,
            pool_dataset=pool_dataset,
            train_dataset=train_dataset,
            key=subkey,
        )

        # retrain the model
        key, subkey = random.split(key, 2)

        training_generator = NumpyLoader(
            dataset=train_dataset,
            batch_size=64,
            shuffle=True,
        )

        params, _ = fit(
            loss=loss,
            params=pretrained_params,
            eval_model=eval_model,
            optimiser=optimiser,
            num_epochs=100,
            train_generator=training_generator,
            validation_generator=validation_generator,
            key=subkey,
        )

        # obtain test accuracy
        test_accuracy_history.append(
            compute_model_accuracy(
                partial(eval_model.apply, params),
                test_generator,
            )
        )

    return jnp.array(test_accuracy_history)
