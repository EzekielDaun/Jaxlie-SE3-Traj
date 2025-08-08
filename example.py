import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxlie import SE3

from example_utils import render_trajectory
from jaxlie_se3_traj import (
    MultiplyDirection,
    SE3DeltaLog,
    SE3DeltaTiltTorsionRolling,
    SE3DeltaTiltTorsionSpiral,
    SE3Trajectory,
)
from jaxlie_se3_traj.bump_method import (
    cubic,
    linear,
    quintic,
    seven_seg_s_curve,
    trig_cos_ease,
    trig_sin_integral,
)

if __name__ == "__main__":
    trajectory = SE3Trajectory.from_deltas(
        (
            SE3DeltaTiltTorsionSpiral(
                MultiplyDirection.RIGHT,
                8.0,
                Partial(seven_seg_s_curve, m=1.5, rho=1 / 2),
                jnp.pi / 2,
                2,
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                cubic,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, 0, jnp.pi / 2])),
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                cubic,
                tuple(jnp.array([0.0, 0.0, 0.5, 0, 0, 0])),
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                linear,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, tilt_angle := jnp.pi / 2, 0])),
            ),
            SE3DeltaTiltTorsionRolling(
                MultiplyDirection.RIGHT,
                3.0,
                trig_sin_integral,
                tilt_angle=tilt_angle,
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                quintic,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, -tilt_angle, 0])),
            ),
            SE3DeltaLog(
                MultiplyDirection.RIGHT,
                1.0,
                trig_cos_ease,
                tuple(jnp.array([0.0, 0.0, -0.5, 0, 0, 0])),
            ),
        ),
    )

    trajectory_f = jax.jit(trajectory)
    # Above cannot be JIT-ed, comment out the next line to see.
    trajectory_f = jax.jit(trajectory.pose_fn)  # This can be JIT-ed.

    @jax.jit
    def offset_traj(t):
        return SE3.from_translation(jnp.array([-0.5, -0.5, 0.0])) @ trajectory_f(t)

    for t in jnp.linspace(0.0, trajectory.total_time, num=10):
        print(f"t={t:.2f}")
        print(f"T={offset_traj(t)}")
        print(f"dT={jax.jacobian(offset_traj)(t)}")

    pose_batched = jax.vmap(offset_traj)(
        jnp.linspace(0.0, trajectory.total_time, num=10)
    )
    print(f"{pose_batched=}")

    render_trajectory(
        lambda t: SE3.from_translation(jnp.array([0, 0, 0.5])) @ offset_traj(t),
        t_range=(0.0, trajectory.total_time),
    )
