from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxlie import SE3
from jaxtyping import Float

from .se3_delta_segments import MultiplyDirection, SE3DeltaBase


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


@pytree_dataclass(frozen=True)
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
