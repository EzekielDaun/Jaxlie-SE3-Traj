from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax_dataclasses import pytree_dataclass
from jaxlie import SE3, SO3
from jaxtyping import Float

from .bump_method import mirror_mapping


class MultiplyDirection(Enum):
    RIGHT = auto()
    LEFT = auto()


@pytree_dataclass(frozen=True)
class SE3DeltaBase(ABC):
    direction: MultiplyDirection
    duration: float
    bump_function: Callable[[Float], Float]

    @abstractmethod
    def compile(self) -> Callable[[float], SE3]:
        pass


@pytree_dataclass(frozen=True)
class SE3DeltaLog(SE3DeltaBase):
    delta_log: tuple[float, ...]

    def compile(self) -> Callable[[float], SE3]:
        log = jnp.array(self.delta_log)
        bump_fn = self.bump_function
        return lambda s: SE3.exp(bump_fn(s) * log)


@pytree_dataclass(frozen=True)
class SE3DeltaTiltTorsionRolling(SE3DeltaBase):
    tilt_angle: float

    def compile(self) -> Callable[[float], SE3]:
        tilt = self.tilt_angle
        bump_fn = self.bump_function

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


@pytree_dataclass(frozen=True)
class SE3DeltaTiltTorsionSpiral(SE3DeltaBase):
    tilt_angle: float
    turns: int

    def compile(self) -> Callable[[float], SE3]:
        bump_fn = Partial(mirror_mapping, f01=self.bump_function)

        def fn(s):
            return SE3.from_rotation(
                SO3.exp(
                    SO3.from_z_radians(s * 2 * jnp.pi * self.turns).apply(
                        jnp.array([0.0, bump_fn(s) * self.tilt_angle, 0.0])
                    )
                )
            )

        return fn


@jax.jit
def tilt_torsion_twist_SE3(
    torsion_angle_rad: Float, tilt_angle_rad: Float, tilt_direction_rad: Float
) -> SE3:
    return SE3.from_rotation(
        SO3.from_z_radians(torsion_angle_rad)
        @ SO3.exp(
            tilt_angle_rad
            * SO3.from_z_radians(tilt_direction_rad).apply(jnp.array([1.0, 0.0, 0.0]))
        )
    )


@pytree_dataclass(frozen=True)
class SE3DeltaTiltTorsionFullSpace(SE3DeltaBase):
    tilt_angle_amplitude: float
    tilt_angle_frequency: float
    torsion_angle_amplitude: float
    torsion_angle_frequency: float
    tilt_direction_wrap_frequency: float

    def compile(self) -> Callable[[float], SE3]:
        bump_fn = Partial(mirror_mapping, f01=self.bump_function)

        def fn(s):
            tilt_direction_angle = (
                self.tilt_direction_wrap_frequency * 2 * jnp.pi * s * self.duration / 2
            )

            bumped_s = bump_fn(s)

            tilt_angle = (
                self.tilt_angle_amplitude
                * (
                    jnp.cos(
                        self.tilt_angle_frequency
                        * 2
                        * jnp.pi
                        * bumped_s
                        * self.duration
                        / 2
                    )
                    - 1
                )
                / -2
            )
            torsion_angle = self.torsion_angle_amplitude * jnp.sin(
                self.torsion_angle_frequency * 2 * jnp.pi * bumped_s * self.duration / 2
            )
            return tilt_torsion_twist_SE3(
                torsion_angle, tilt_angle, tilt_direction_angle
            )

        return jax.jit(fn)
