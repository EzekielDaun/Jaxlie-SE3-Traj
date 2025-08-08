from typing import Callable

import jax.numpy as jnp
from jaxtyping import Float

from .seven_seg_s_curve import seven_seg_s_curve  # noqa: F401


def linear(s: Float) -> Float:
    return jnp.clip(s, 0.0, 1.0)


def cubic(s: Float) -> Float:
    return jnp.where(s <= 0.0, 0.0, jnp.where(s >= 1.0, 1.0, 3 * s**2 - 2 * s**3))  # type: ignore


def quintic(s: Float) -> Float:
    return jnp.where(
        s <= 0.0, 0.0, jnp.where(s >= 1.0, 1.0, s**3 * (10 - 15 * s + 6 * s**2))  # type: ignore
    )


def trig_cos_ease(s: Float) -> Float:
    return jnp.where(
        s <= 0.0, 0.0, jnp.where(s >= 1.0, 1.0, 0.5 - 0.5 * jnp.cos(jnp.pi * s))
    )


def trig_sin_integral(s: Float) -> Float:
    return jnp.where(
        s <= 0.0,
        0.0,
        jnp.where(s >= 1.0, 1.0, s - jnp.sin(2 * jnp.pi * s) / (2 * jnp.pi)),  # type: ignore
    )


def mirror_mapping(t: Float, f01: Callable[[Float], Float]) -> Float:
    return jnp.where(
        t <= 0.5,
        f01(2 * t),
        f01(2 * (1 - t)),
    )
