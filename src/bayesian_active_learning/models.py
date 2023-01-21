from functools import partial
from typing import Callable, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
from jax import nn, random


class BayesianConvNet(hk.Module):
    "A simple Bayesian convolutional network"

    def __init__(
        self,
        num_classes: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        # dropout_rates: Tuple[float, float],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        # self.dropout_rates = dropout_rates

    def __call__(
        self,
        dropout_rates: Tuple[float, float],
        input: jnp.ndarray,
    ) -> jnp.ndarray:
        out = input

        out = hk.Conv2D(output_channels=32, kernel_shape=4)(out)
        out = self.activation(out)
        out = hk.Conv2D(output_channels=32, kernel_shape=4)(out)
        out = self.activation(out)
        out = hk.MaxPool(window_shape=2, strides=1, padding="VALID")(out)
        if dropout_rates[0] != 0:
            out = hk.dropout(hk.next_rng_key(), dropout_rates[0], out)
        flat = hk.Flatten()
        out = flat(out)
        out = hk.Linear(output_size=128)(out)
        out = self.activation(out)
        if dropout_rates[1] != 0:
            out = hk.dropout(hk.next_rng_key(), dropout_rates[1], out)
        logits = hk.Linear(output_size=self.num_classes)(out)

        return logits


def define_model(
    num_classes: int,
    dropout_rates: Tuple[float, float],
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    net = BayesianConvNet(
        num_classes=num_classes,
        activation=activation,
    )
    return partial(net, dropout_rates)
