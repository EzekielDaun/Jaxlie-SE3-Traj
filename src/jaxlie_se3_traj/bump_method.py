from enum import Enum, auto
from typing import Callable

import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Float

from .seven_seg_s_curve import seven_seg_s_curve


class BumpMethod(Enum):
    CUBIC = auto()
    QUINTIC = auto()
    TRIG_COS_EASE = auto()
    TRIG_SIN_INTEGRAL = auto()
    SEVEN_SEG_S_DEFAULT = auto()
    LINEAR = auto()


def bump_function(method: BumpMethod) -> Callable[[float], float]:
    def cubic(s: float) -> float:
        return jnp.where(s <= 0.0, 0.0, jnp.where(s >= 1.0, 1.0, 3 * s**2 - 2 * s**3))  # type: ignore

    def quintic(s: float) -> float:
        return jnp.where(
            s <= 0.0, 0.0, jnp.where(s >= 1.0, 1.0, s**3 * (10 - 15 * s + 6 * s**2))  # type: ignore
        )

    def trig_cos_ease(s: float) -> float:
        return jnp.where(
            s <= 0.0, 0.0, jnp.where(s >= 1.0, 1.0, 0.5 - 0.5 * jnp.cos(jnp.pi * s))  # type: ignore
        )

    def trig_sin_integral(s: float) -> float:
        return jnp.where(
            s <= 0.0,
            0.0,
            jnp.where(s >= 1.0, 1.0, s - jnp.sin(2 * jnp.pi * s) / (2 * jnp.pi)),  # type: ignore
        )

    if method == BumpMethod.CUBIC:
        return cubic
    elif method == BumpMethod.QUINTIC:
        return quintic
    elif method == BumpMethod.TRIG_COS_EASE:
        return trig_cos_ease
    elif method == BumpMethod.TRIG_SIN_INTEGRAL:
        return trig_sin_integral
    elif method == BumpMethod.SEVEN_SEG_S_DEFAULT:
        return Partial(seven_seg_s_curve, m=1.5, rho=1 / 2)
    elif method == BumpMethod.LINEAR:
        return lambda s: jnp.clip(s, 0.0, 1.0)  # type: ignore
    else:
        raise ValueError(f"Unknown bump method: {method}")


def mirror_mapping(t: Float, f01: Callable[[Float], Float]) -> Float:
    t = jnp.asarray(t)
    return jnp.where(
        t <= 0.5,
        f01(2 * t),
        f01(2 * (1 - t)),
    )
