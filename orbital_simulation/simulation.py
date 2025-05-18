from datetime import timedelta
from itertools import combinations

from pathlib import Path

import astropy
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import toml
from astropy import units as u
from astropy.constants import G
from astropy.time import Time
from tqdm.auto import tqdm

from .rigidbodies import Rigidbody


def gravitational_acceleration(body1: Rigidbody, body2: Rigidbody, G: float = G.value):
    """
    Calculates the gravitational acceleration of body1 towards body2.

    Parameters
    ----------

    body1 : Rigidbody
    The first rigidbody, for which the acceleration will be calculated.

    body2 : Rigidbody
    The second rigidbody.

    G : float, optional
    The gravitational constant which will be used for the calculation.
    Default is ``astropy.constants.G``.

    Returns
    -------

    a : np.ndarray
    The gravitational acceleration of body1 towards body2.

    """
    diff = body1.get_current_position() - body2.get_current_position()
    return -G * (body2.mass) / np.linalg.norm(diff) ** 2 * (diff) / np.linalg.norm(diff)


class Simulation:
    def __init__(
        self,
        dt: float,
        gravitational_constant: float = G.value,
    ):
        """
        Initializes a n-body simulation.

        Parameters
        ----------

        """

        self.rigidbodies = []
        self._dt = dt
        self.G = gravitational_constant

        self.time = 0
        self.n_it = 0

    @classmethod
    def copy(cls, simulation):

        cls = cls()

        cls.rigidbodies = [Rigidbody.copy(body) for body in simulation.rigidbodies]
        cls._dt = simulation._dt
        cls.G = simulation.G

        cls.time = simulation.time
        cls.n_it = simulation.n_it

        return cls

    def get_rigidbody_by_name(self, name: str):
        for body in self.rigidbodies:
            if body.name == name:
                return body

        return None

    def add_rigidbody(self, rigidbody: Rigidbody):
        self.rigidbodies.append(rigidbody)

    def load_planets(
        self,
        time: astropy.time.Time = Time.now(),
        exclude: list[str] = [],
        cutoff_index: int = None,
        include_moon: bool = True,
        config: str or None = None,
    ):

        if config is None:
            config = str(Path(__file__).with_name("default_planets.toml"))

        for key in list(toml.load(config).keys())[:cutoff_index]:

            if not include_moon:
                exclude.append("moon")

            if key not in exclude:
                self.add_rigidbody(
                    Rigidbody.from_name(name=key, time=time, config=config)
                )

    def _get_max_length(self):
        return np.max(
            np.abs(
                [
                    [
                        body.get_max_pos(coord) - body.get_min_pos(coord)
                        for coord in np.arange(0, 3)
                    ]
                    for body in self.rigidbodies
                ]
            )
        )

    def run(self, time: float):
        for t in tqdm(
            np.arange(start=self.time, stop=self.time + time + 1, step=self._dt),
            desc="Simulating steps",
        ):
            self.time = t
            self.n_it += 1
            self.step()

    def reset(self):
        self.time = 0
        self.n_it = 0
        for body in self.rigidbodies:
            body.reset()

    def info(self):
        for body in self.rigidbodies:
            body.info()

    def step(self):

        for body in self.rigidbodies:
            body.acceleration = np.array((0, 0, 0))

        for body1, body2 in combinations(self.rigidbodies, 2):
            if (
                body1 == body2
                or body2 in body1.exclude_bodies
                or body1 in body2.exclude_bodies
            ):
                continue

            body2.acceleration = body2.acceleration + gravitational_acceleration(
                body2, body1, G=self.G
            )
            body1.acceleration = body1.acceleration + gravitational_acceleration(
                body1, body2, G=self.G
            )

        for body in self.rigidbodies:
            body.move(self._dt)

    def _configure_axis(
        self,
        fig: matplotlib.figure.Figure,
        ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
        log: bool,
        center_body: Rigidbody,
        center_view: bool,
        zoom_factor: float,
        zoom_center: Rigidbody,
        aspect_scale: tuple[float],
        view_param: dict,
        unit: astropy.units.Unit,
        to_scale: bool,
        vax: matplotlib.axes.Axes or None = None,
        plot_velocity: list[Rigidbody] = [],
    ):

        ax.set_xlabel(f"Relative $x$ in {str(unit)}")
        ax.set_ylabel(f"Relative $y$ in {str(unit)}")
        ax.set_zlabel(f"Relative $z$ in {str(unit)}")

        if log:
            ax.set_xscale("symlog")
            ax.set_yscale("symlog")
            ax.set_zscale("symlog")

        if zoom_center is None:
            self._set_axis_limits(
                fig,
                ax,
                center_body,
                center_view,
                zoom_factor,
                np.zeros(3),
                unit=unit,
                vax=vax,
                plot_velocity=plot_velocity,
            )

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        ax.set_box_aspect(
            (
                (xlim[1] - xlim[0]) * aspect_scale[0],
                (ylim[1] - ylim[0]) * aspect_scale[1],
                (zlim[1] - zlim[0]) * aspect_scale[2],
            )
        )

        if not to_scale:
            ax.legend(
                loc="upper left",
                ncols=np.max([len(self.rigidbodies) // 3, 1]),
                fontsize="small"
            )

        ax.view_init(**view_param)

        if vax is not None:
            vax.set_xlabel("Time in s")
            vax.set_ylabel("Relative orbital velocity in m/s")

        return fig, ax, vax

    def _set_axis_limits(
        self,
        fig: matplotlib.figure.Figure,
        ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
        center_body: Rigidbody,
        center_view: bool,
        zoom_factor: float,
        zoom_center: Rigidbody,
        unit: astropy.units.Unit,
        vax: matplotlib.axes.Axes or None = None,
        plot_velocity: list[Rigidbody] = [],
    ):

        limits = [None, None, None]

        if not center_view:
            for i in np.arange(3):
                limits[i] = (
                    (
                        np.min(
                            [
                                body.get_min_pos(i, center_body=center_body)
                                for body in self.rigidbodies
                            ]
                        )
                    )
                    / zoom_factor,
                    (
                        np.max(
                            [
                                body.get_max_pos(i, center_body=center_body)
                                for body in self.rigidbodies
                            ]
                        )
                    )
                    / zoom_factor,
                )
        else:
            for i in np.arange(3):
                maximum = (
                    np.max(
                        [
                            np.abs(
                                [
                                    body.get_max_pos(i, center_body=center_body),
                                    body.get_min_pos(i, center_body=center_body),
                                ]
                            )
                            for body in self.rigidbodies
                        ]
                    )
                ) / zoom_factor

                limits[i] = (-maximum, maximum)

        methods = [ax.set_xlim, ax.set_ylim, ax.set_zlim]

        for i in np.arange(3):
            if limits[i][0] != limits[i][1]:
                methods[i](
                    ((limits[i][0] + zoom_center[i]) * u.meter).to(unit).value,
                    ((limits[i][1] + zoom_center[i]) * u.meter).to(unit).value,
                )
            else:
                methods[i](
                    -1 / zoom_factor + zoom_center[i], 1 / zoom_factor + zoom_center[i]
                )

        if vax is not None:
            vax.set_ylim(
                0,
                np.max(
                    [
                        body.get_abs_velocities(relative_to=center_body)
                        for body in plot_velocity
                    ]
                )
                * 1.1,
            )

    def animate(
        self,
        steps_per_frame: int,
        framesep: int,
        save_file: str = "animation.mp4",
        fade: int or None = None,
        log=False,
        unit=u.meter,
        view_param=dict(elev=20, azim=-45, roll=0),
        center_body: Rigidbody = None,
        center_view: bool = False,
        aspect_scale: tuple[float] = (1.0, 1.0, 1.0),
        zoom_factor: float = 1.0,
        zoom_center: Rigidbody or None = None,
        to_scale: bool = False,
        draw_acceleration: bool = False,
        plot_velocity: list[Rigidbody] = [],
        dpi="figure",
    ):

        frames = self.n_it // steps_per_frame

        if len(plot_velocity) == 0:
            fig = plt.figure(layout="constrained")
            ax = fig.add_subplot(projection="3d")
            vax = None
        else:
            fig = plt.figure(layout="constrained")
            ax = fig.add_subplot(4, 1, (1, 3), projection="3d")
            vax = fig.add_subplot(4, 1, 4)

        line_plots = [
            ax.plot(
                xs=[],
                ys=[],
                zs=[],
                color=body.trail_color,
                alpha=body.trail_alpha,
                zorder=1,
            )[0]
            for body in self.rigidbodies
        ]
        scatter_plots = [
            ax.scatter(
                [],
                [],
                [],
                color=body.body_color,
                marker=body.marker,
                zorder=2,
                label=body.name,
                s=body.get_marker_size(),
                alpha=body.body_alpha,
            )
            for body in self.rigidbodies
        ]
        if draw_acceleration:
            quiver_plots = [
                ax.quiver(
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    color=body.body_color,
                    zorder=1,
                )
                for body in self.rigidbodies
            ]

        vel_plots = [
            vax.plot(
                np.arange(self._dt * (self.n_it + 1), step=self._dt),
                np.zeros_like(np.arange(self._dt * (self.n_it + 1), step=self._dt)),
                color=body.body_color,
            )[0]
            for body in plot_velocity
        ]

        def update(frame):

            frame += 1
            frame = np.min([frame * steps_per_frame, self.n_it])

            fig.suptitle(
                f"Simulation time: {str(timedelta(seconds=int(self._dt * frame)))} "
                f"| Step: {frame}"
            )

            fade_frames = fade if fade is not None else frame

            for idx, body in zip(np.arange(len(self.rigidbodies)), self.rigidbodies):

                if center_body is None:
                    positions = (
                        (
                            np.reshape(body.positions, (-1, 3))[
                                np.max([0, frame - fade_frames]) : frame
                            ]
                            * u.meter
                        )
                        .to(unit)
                        .value
                    )
                    current_position = positions[-1]

                else:
                    positions = (
                        (
                            np.reshape(body.positions - center_body.positions, (-1, 3))[
                                np.max([0, frame - fade_frames]) : frame
                            ]
                            * u.meter
                        )
                        .to(unit)
                        .value
                    )
                    accelerations = np.reshape(body.accelerations, (-1, 3))[
                        np.max([0, frame - fade_frames]) : frame
                    ]
                    current_position = positions[-1]
                    current_acceleration = accelerations[-1]

                line_plots[idx].set_data_3d(
                    positions[:, 0], positions[:, 1], positions[:, 2]
                )
                scatter_plots[idx].remove()
                scatter_plots[idx] = ax.scatter(
                    *current_position,
                    color=body.body_color,
                    marker=body.marker,
                    zorder=2,
                    label=body.name,
                    s=body.get_marker_size(
                        to_scale=to_scale, max_length=self._get_max_length(), unit=unit
                    ),
                    alpha=body.body_alpha,
                )

                if draw_acceleration:
                    quiver_plots[idx].remove()
                    quiver_plots[idx] = ax.quiver(
                        *current_position,
                        *(current_acceleration * (self._get_max_length() * 1e-2)),
                        color=body.body_color,
                        zorder=1,
                    )

            for idx, body in zip(np.arange(len(plot_velocity)), plot_velocity):
                velocities = body.get_abs_velocities(relative_to=center_body)
                velocities[frame:] = np.nan

                vel_plots[idx].set_xdata(
                    np.arange(start=0, stop=self._dt * (self.n_it + 1), step=self._dt)
                )
                vel_plots[idx].set_ydata(velocities)

            if zoom_center is not None:
                positions = (
                    (
                        np.reshape(zoom_center.positions, (-1, 3))[
                            np.max([0, frame - fade_frames]) : frame
                        ]
                        * u.meter
                    )
                    .to(unit)
                    .value
                )
                current_position = positions[-1]
                self._set_axis_limits(
                    fig,
                    ax,
                    center_body,
                    center_view,
                    zoom_factor,
                    current_position,
                    unit=unit,
                    vax=vax,
                    plot_velocity=plot_velocity,
                )

        fig, ax, vax = self._configure_axis(
            fig=fig,
            ax=ax,
            log=log,
            center_body=center_body,
            center_view=center_view,
            zoom_factor=zoom_factor,
            zoom_center=zoom_center,
            aspect_scale=aspect_scale,
            view_param=view_param,
            unit=unit,
            to_scale=to_scale,
            vax=vax,
            plot_velocity=plot_velocity,
        )

        writer = None
        if save_file[-4:].lower() == "gif":
            writer = animation.PillowWriter(
                fps=1 / (framesep * 1e-3),
            )
            writer.setup(fig=fig, outfile=save_file, dpi=dpi)

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=frames, interval=framesep
        )

        def progress_func(_i, _n):
            progress_bar.update(1)

        with tqdm(total=frames, desc="Saving animation") as progress_bar:
            if writer is None:
                ani.save(save_file, progress_callback=progress_func, dpi=dpi)
            else:
                ani.save(save_file, writer=writer, progress_callback=progress_func, dpi=dpi)

        return fig, ax

    def plot_state(
        self,
        log: bool = False,
        unit: astropy.units.Unit = u.meter,
        view_param: dict = dict(elev=20, azim=-45, roll=0),
        center_body: Rigidbody = None,
        center_view: bool = False,
        aspect_scale: tuple[float] = (1.0, 1.0, 1.0),
        zoom_factor: float = 1.0,
        zoom_center: Rigidbody = None,
        save_file: str or None = None,
        to_scale: bool = False,
    ):
        fig = plt.figure(figsize=(7, 7), layout="constrained")
        ax = fig.add_subplot(projection="3d")

        fig.suptitle(
            f"Simulation time: {timedelta(seconds=int(self.time))} | Step: {self.n_it}"
        )

        for body in self.rigidbodies:

            if center_body is None:
                positions = (
                    (np.reshape(body.positions, (-1, 3)) * u.meter).to(unit).value
                )
                current_position = (
                    (body.get_current_position() * u.meter).to(unit).value
                )

            else:
                positions = (
                    (
                        np.reshape(body.positions - center_body.positions, (-1, 3))
                        * u.meter
                    )
                    .to(unit)
                    .value
                )
                current_position = (
                    (
                        (
                            body.get_current_position()
                            - center_body.get_current_position()
                        )
                        * u.meter
                    )
                    .to(unit)
                    .value
                )

            ax.scatter(
                *current_position,
                color=body.body_color,
                alpha=body.body_alpha,
                marker=body.marker,
                zorder=2,
                label=body.name,
                s=body.get_marker_size(
                    to_scale=to_scale, max_length=self._get_max_length(), unit=unit
                ),
            )

            ax.plot(
                xs=positions[:, 0],
                ys=positions[:, 1],
                zs=positions[:, 2],
                color=body.trail_color,
                alpha=body.trail_alpha,
                zorder=1,
            )

        fig, ax, vax = self._configure_axis(
            fig=fig,
            ax=ax,
            log=log,
            center_body=center_body,
            center_view=center_view,
            zoom_factor=zoom_factor,
            zoom_center=zoom_center,
            aspect_scale=aspect_scale,
            view_param=view_param,
            unit=unit,
            to_scale=to_scale,
        )

        if zoom_center is not None:
            self._set_axis_limits(
                fig,
                ax,
                center_body,
                center_view,
                zoom_factor,
                zoom_center.get_current_position(),
                unit=unit,
            )

        if save_file is not None:
            fig.savefig(save_file)

        return fig, ax
