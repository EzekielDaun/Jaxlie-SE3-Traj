from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxlie import SE3, SO3
from jaxtyping import Float


class BumpMethod(Enum):
    CUBIC = auto()
    QUINTIC = auto()
    TRIG_COS_EASE = auto()
    TRIG_SIN_INTEGRAL = auto()
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
    elif method == BumpMethod.LINEAR:
        return lambda s: jnp.clip(s, 0.0, 1.0)  # type: ignore
    else:
        raise ValueError(f"Unknown bump method: {method}")


class MultiplyDirection(Enum):
    RIGHT = auto()
    LEFT = auto()


@jdc.pytree_dataclass(frozen=True)
class SE3DeltaBase(ABC):
    direction: MultiplyDirection
    duration: float
    bump_method: BumpMethod

    @abstractmethod
    def compile(self) -> Callable[[float], SE3]:
        pass


@jdc.pytree_dataclass(frozen=True)
class SE3DeltaLog(SE3DeltaBase):
    delta_log: tuple[float, ...]

    def compile(self) -> Callable[[float], SE3]:
        log = jnp.array(self.delta_log)
        bump_fn = bump_function(self.bump_method)
        return lambda s: SE3.exp(bump_fn(s) * log)


@jdc.pytree_dataclass(frozen=True)
class SE3DeltaTiltTorsionRolling(SE3DeltaBase):
    tilt_angle: float

    def compile(self) -> Callable[[float], SE3]:
        tilt = self.tilt_angle
        bump_fn = bump_function(self.bump_method)

        def fn(s):
            return SE3.from_rotation(
                SO3.from_y_radians(tilt).inverse()
                @ SO3.exp(
                    tilt
                    * SO3.from_z_radians(bump_fn(s) * 2 * jnp.pi).apply(
                        jnp.array([0.0, 1.0, 0.0])
                    )
                )
            )

        return fn


@jdc.pytree_dataclass(frozen=True)
class SE3DeltaTiltTorsionSpiral(SE3DeltaBase):
    tilt_angle: float
    turns: int

    def compile(self) -> Callable[[float], SE3]:
        def mirror_mapping(t: Float, f01: Callable[[Float], Float]) -> Float:
            t = jnp.asarray(t)
            return jnp.where(
                t <= 0.5,
                f01(2 * t),
                f01(2 * (1 - t)),
            )

        bump_fn = partial(mirror_mapping, f01=bump_function(self.bump_method))

        def fn(s):
            return SE3.from_rotation(
                SO3.exp(
                    SO3.from_z_radians(s * 2 * jnp.pi * self.turns).apply(
                        jnp.array([0.0, bump_fn(s) * self.tilt_angle, 0.0])
                    )
                )
            )

        return fn


# directions: 0 = RIGHT, 1 = LEFT
@partial(jax.jit, static_argnames=("compiled_deltas",))
def evaluate_se3_trajectory(
    compiled_deltas: tuple[Callable[[float], SE3], ...],
    start_times: Float,
    durations: Float,
    directions: Float,
    t: float,
) -> SE3:
    s_values = jnp.clip((t - start_times) / durations, 0.0, 1.0)
    delta_Ts = tuple(fn(s) for fn, s in zip(compiled_deltas, s_values))

    T = SE3.identity()
    for dT, direction in zip(reversed(delta_Ts), reversed(directions)):
        T = jax.lax.cond(
            direction == 0,
            lambda _: T @ dT,
            lambda _: dT @ T,
            operand=None,
        )
    return T


@jdc.pytree_dataclass(frozen=True)
class SE3Trajectory:
    compiled_deltas: tuple[Callable[[float], SE3], ...]
    durations: Float
    start_times: Float
    end_times: Float
    total_time: float
    directions: Float

    @classmethod
    def from_deltas(cls, deltas: tuple[SE3DeltaBase, ...]) -> "SE3Trajectory":
        compiled = tuple(d.compile() for d in deltas)
        durations = jnp.array([d.duration for d in deltas])
        end_times = jnp.cumsum(durations)
        start_times = jnp.concatenate([jnp.array([0.0]), end_times[:-1]])
        total_time: float = end_times[-1]  # type: ignore
        directions = jnp.array(
            [0 if d.direction == MultiplyDirection.RIGHT else 1 for d in deltas]
        )
        return cls(compiled, durations, start_times, end_times, total_time, directions)

    def __call__(self, t: float) -> SE3:
        return evaluate_se3_trajectory(
            self.compiled_deltas,
            self.start_times,
            self.durations,
            self.directions,
            t,
        )
