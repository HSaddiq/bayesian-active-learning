from typing import Callable

import jax.numpy as jnp
from torch.utils.data import DataLoader


def compute_single_batch_accuracy(
    logits: jnp.ndarray, labels: jnp.ndarray
) -> jnp.ndarray:
    """Returns the accuracy for one batch

    Args:
        logits (jnp.ndarray): logits from the model
        labels (jnp.ndarray): true label

    Returns:
        jnp.ndarray: accuracy for the batch
    """
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(jnp.argmax(labels, axis=-1) == preds)


def compute_model_accuracy(
    eval_model: Callable[[jnp.ndarray], jnp.ndarray], dataloader: DataLoader
) -> float:
    """Computes the accuracy of the model on a given dataloader

    Args:
        eval_model (Callable[[jnp.ndarray], jnp.ndarray]): _description_
        dataloader (DataLoader): dataloader containing

    Returns:
        float: accuracy of the model
    """
    correct_pred_counts = dataset_size = 0

    for xs, labels in dataloader:
        batch_size = xs.shape[0]
        logits = eval_model(xs)
        batch_correct_counts = (
            compute_single_batch_accuracy(logits, labels) * batch_size
        )
        correct_pred_counts += batch_correct_counts
        dataset_size += batch_size

    return correct_pred_counts / dataset_size
