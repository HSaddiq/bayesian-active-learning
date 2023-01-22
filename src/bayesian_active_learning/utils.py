import jax.numpy as jnp


def one_hot(x: jnp.ndarray, k: float, dtype=jnp.float32) -> jnp.ndarray:
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)
