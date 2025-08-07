import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxlie import SE3

from jaxlie_se3_traj import (
    BumpMethod,
    MultiplyDirection,
    SE3DeltaLog,
    SE3DeltaTiltTorsionRolling,
    SE3DeltaTiltTorsionSpiral,
    SE3Trajectory,
)

if __name__ == "__main__":

    def plot_se3(ax, T: SE3, scale: float = 0.5):
        origin = T.translation()
        R = T.rotation().as_matrix()

        colors = ["r", "g", "b"]
        for i in range(3):
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                R[0, i],
                R[1, i],
                R[2, i],
                length=scale,
                color=colors[i],
            )

    def render_trajectory(trajectory, t_range, fps=24):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(221, projection="3d")
        ax1.view_init(elev=30, azim=-60)  # type: ignore
        ax2 = fig.add_subplot(222, projection="3d")
        ax2.view_init(elev=90, azim=-90)  # type: ignore
        ax3 = fig.add_subplot(223, projection="3d")
        ax3.view_init(elev=0, azim=-90)  # type: ignore
        ax4 = fig.add_subplot(224, projection="3d")
        ax4.view_init(elev=0, azim=0)  # type: ignore

        times = np.linspace(*t_range, num=int((t_range[1] - t_range[0]) * fps))
        positions = []

        for t in times:
            T = trajectory(t)
            positions.append(T.translation())

        positions = np.stack(positions)
        size = np.max(np.abs([positions.min(), positions.max()]))

        for t in times:
            for ax in (ax1, ax2, ax3, ax4):
                ax.clear()
                ax.set_xlim(-size, size)
                ax.set_ylim(-size, size)
                ax.set_zlim(-size, size)  # type: ignore
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")  # type: ignore

                T = trajectory(t)
                plot_se3(ax, T)

            fig.canvas.draw()
            fig.tight_layout()
            plt.pause(1 / fps)

        plt.close(fig)

    trajectory = SE3Trajectory.from_deltas(
        (
            SE3DeltaTiltTorsionSpiral(
                MultiplyDirection.RIGHT,
                8.0,
                BumpMethod.TRIG_SIN_INTEGRAL,
                jnp.pi / 2,
                2,
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                BumpMethod.CUBIC,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, 0, jnp.pi / 2])),
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                BumpMethod.CUBIC,
                tuple(jnp.array([0.0, 0.0, 0.5, 0, 0, 0])),
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                BumpMethod.LINEAR,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, tilt_angle := jnp.pi / 2, 0])),
            ),
            SE3DeltaTiltTorsionRolling(
                MultiplyDirection.RIGHT,
                3.0,
                BumpMethod.TRIG_SIN_INTEGRAL,
                tilt_angle=tilt_angle,
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                BumpMethod.QUINTIC,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, -tilt_angle, 0])),
            ),
            SE3DeltaLog(
                MultiplyDirection.RIGHT,
                1.0,
                BumpMethod.TRIG_COS_EASE,
                tuple(jnp.array([0.0, 0.0, -0.5, 0, 0, 0])),
            ),
        ),
    )

    for t in jnp.linspace(0.0, trajectory.total_time, num=10):
        print(f"t={t:.2f}, T={trajectory(t)}")

    render_trajectory(
        lambda t: SE3.from_translation(jnp.array([0, 0, 0.5])) @ trajectory(t),
        t_range=(0.0, trajectory.total_time),
    )
