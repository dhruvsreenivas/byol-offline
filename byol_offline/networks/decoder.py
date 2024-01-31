import chex
import haiku as hk
import jax
import jax.numpy as jnp

"""Various decoder modules."""


class DrQv2Decoder(hk.Module):
    """DrQv2 decoder. Reverse of DrQv2 encoder."""

    def __init__(self, in_channel: int):
        super().__init__()

        self._initializer = hk.initializers.Orthogonal(scale=jnp.sqrt(2))
        self._convnet = hk.Sequential(
            [
                hk.Conv2DTranspose(
                    32,
                    kernel_shape=3,
                    stride=1,
                    padding="VALID",
                    w_init=self._initializer,
                ),
                jax.nn.relu,
                hk.Conv2DTranspose(
                    32,
                    kernel_shape=3,
                    stride=1,
                    padding="VALID",
                    w_init=self._initializer,
                ),
                jax.nn.relu,
                hk.Conv2DTranspose(
                    32,
                    kernel_shape=3,
                    stride=1,
                    padding="VALID",
                    w_init=self._initializer,
                ),
                jax.nn.relu,
                hk.Conv2DTranspose(
                    in_channel,
                    kernel_shape=3,
                    stride=2,
                    padding="VALID",
                    w_init=self._initializer,
                ),
            ]
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        out_dim = 20000

        # first preprocess
        x = hk.Linear(out_dim, w_init=self._initializer)(x)
        x = jnp.reshape(x, (-1, 25, 25, 32))

        # return conv output
        return self._convnet(x)


class DreamerDecoder(hk.Module):
    """DreamerV2 decoder."""

    def __init__(self, in_channel: int, depth: int):
        super().__init__()
        self._depth = depth

        self._initializer = hk.initializers.VarianceScaling(
            1.0, "fan_avg", distribution="uniform"
        )
        self._convnet = hk.Sequential(
            [
                hk.Conv2DTranspose(
                    depth * 4,
                    kernel_shape=5,
                    stride=2,
                    padding="VALID",
                    w_init=self._initializer,
                ),
                jax.nn.elu,
                hk.Conv2DTranspose(
                    depth * 2,
                    kernel_shape=5,
                    stride=2,
                    padding="VALID",
                    w_init=self._initializer,
                ),
                jax.nn.elu,
                hk.Conv2DTranspose(
                    depth,
                    kernel_shape=6,
                    stride=2,
                    padding="VALID",
                    w_init=self._initializer,
                ),
                jax.nn.elu,
                hk.Conv2DTranspose(
                    in_channel,
                    kernel_shape=6,
                    stride=2,
                    padding="VALID",
                    w_init=self._initializer,
                ),
            ]
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        x = hk.Linear(32 * self._depth, w_init=self._initializer)(x)
        x = jnp.reshape(x, (-1, 1, 1, 32 * self._depth))
        mean = self._convnet(x)
        return mean
