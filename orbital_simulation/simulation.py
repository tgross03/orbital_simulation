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
    Default is the numerical value of ``astropy.constants.G``.

    Returns
    -------

    a : np.ndarray
    The gravitational acceleration of body1 towards body2 in meters / second**2.

    """
    diff = body1.get_current_position() - body2.get_current_position()
    return -G * (body2.mass) / np.linalg.norm(diff) ** 2 * (diff) / np.linalg.norm(diff)


class Simulation:
    def __init__(
        self,
        dt: float,
        bodies: list = [],
        gravitational_constant: float = G.value,
    ):
        """
        Initializes a new n-body simulation.

        Parameters
        ----------

        dt : float
        The time difference between every simulation.

        bodies : list[Rigidbody], optional
        A list of rigidbodies to add to the simulation.

        G : float, optional
        The gravitational constant which will be used for the calculation.
        Default is the numerical value of ``astropy.constants.G``.

        """

        self.rigidbodies = []

        for body in bodies:
            self.add_rigidbody(body)

        self._dt = dt
        self.G = gravitational_constant

        self.time = 0
        self.n_it = 0

    @classmethod
    def copy(cls, simulation: "Simulation"):
        """
        Creates a deep copy of a simulation instance.

        Parameters
        ----------

        simulation: Simulation
        The simulation to copy.

        """

        cls = cls()

        cls.rigidbodies = [Rigidbody.copy(body) for body in simulation.rigidbodies]
        cls._dt = simulation._dt
        cls.G = simulation.G

        cls.time = simulation.time
        cls.n_it = simulation.n_it

        return cls

    def get_rigidbody_by_name(self, name: str):
        """
        Gets the rigidbody from the simulation with the given name.

        Parameters
        ----------

        name : str
        The name of the wanted rigidbody (case sensitive!)

        Returns
        -------

        body : Rigidbody
        The body with the given name or `None` if no rigidbody with this
        name has been added to the simulation.

        """

        for body in self.rigidbodies:
            if body.name == name:
                return body

        raise KeyError(
            "There was no rigidbody with this name present in this simulation."
        )

    def add_rigidbody(self, rigidbody: Rigidbody):
        """
        Adds a rigidbody to the simulation.

        Parameters
        ----------

        rigidbody : Rigidbody
        The rigidbody to add to the simulation.

        """

        if rigidbody.name in [body.name for body in self.rigidbodies]:
            raise KeyError(
                "This body is already added to this Simulation! "
                "Rigidbody names have to be unique!"
            )

        self.rigidbodies.append(rigidbody)

    def load_planets(
        self,
        time: astropy.time.Time = Time.now(),
        exclude: list[str] = [],
        cutoff_index: int = None,
        include_moon: bool = True,
        config: str or None = None,
    ):
        """
        Loads the all the planets (rigidbodies) from the given config.

        Parameters
        ----------

        time : astropy.time.Time, optional
        The time for which the positions and velocities of the rigidbodies
        should be loaded.
        Default is current time.

        exclude : list[str], optional
        A list of names of rigidbodies to exclude from the simulation
        (e.g. `Sun` or `Jupiter`)

        cutoff_index : int or None, optional
        The index to which the rigidbodies should be added from the config.
        The order of the rigidbodies is determined by their order in the config.
        Default is `None`, meaning the whole list is added.

        include_moon : bool, optional
        Wether to include the earth's moon in the simulation.
        Default is `True`.

        config : str or None, optional
        The path to the config. Default is `None`, meaning that the pre-set
        config from the package will be used.

        """

        if config is None:
            config = str(Path(__file__).with_name("default_planets.toml"))

        exclude = [name.lower() for name in exclude]

        for key in list(toml.load(config).keys())[:cutoff_index]:
            if not include_moon:
                exclude.append("moon")

            if key.lower() not in exclude:
                self.add_rigidbody(
                    Rigidbody.from_name(name=key, time=time, config=config)
                )

    def _get_max_length(self):
        """
        Gets the farthest difference between two points from any rigidbody's trajectory.
        """
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
        """
        Runs the simulation for a given time.

        Parameters
        ----------

        time : float
        The time to run the simulation for in seconds.
        This can be arbitrarily often, meaning that it is possible to
        run the simulation for 30 seconds, edit some things (e.g. change velocities
        or accelerations) and then run it again for 45 seconds.
        The trajectories of the first run will then be continued.

        """
        for t in tqdm(
            np.arange(start=self.time, stop=self.time + time + 1, step=self._dt),
            desc="Simulating steps",
        ):
            self.time = t
            self.n_it += 1
            self._step()

    def reset(self):
        """
        Reset the simulation to its initial state.
        WARNING: This will delete all your previously simulated trajectories!
        """
        self.time = 0
        self.n_it = 0
        for body in self.rigidbodies:
            body.reset()

    def info(self):
        """
        Prints all the information about the simulation and the contained rigidbodies.
        """
        print(f"======== {len(self.rigidbodies)}-body Simulation =========")
        print(self.__dict__)

        for body in self.rigidbodies:
            body.info()

        print("===========================")

    def _step(self):
        """
        Performs all calculations for the current timestep and
        moves all rigidbodies according to their updated accelerations and velocities.
        """

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
        ax: mpl_toolkits.mplot3d.axes3d.Axes3D or matplotlib.axes.Axes,
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
        legend: str = "axes",
    ):
        """
        Configures the axes properties for the plots.
        """

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
            if legend == "axes":
                ax.legend(
                    loc="upper left",
                    ncols=np.max([len(self.rigidbodies) // 3, 1]),
                    fontsize="small",
                )
            elif legend == "fig":
                fig.legend(
                    bbox_to_anchor=(0.35, 0.95),
                    frameon=True,
                    ncols=np.max([len(self.rigidbodies) // 3, 1]),
                    fontsize="small",
                )

        ax.view_init(**view_param)

        if vax is not None:
            vax.set_xlabel("Time in s")
            vax.set_ylabel("Rel. orbital velocity in m/s")

        return fig, ax, vax

    def _set_axis_limits(
        self,
        fig: matplotlib.figure.Figure,
        ax: mpl_toolkits.mplot3d.axes3d.Axes3D or matplotlib.axes.Axes,
        center_body: Rigidbody,
        center_view: bool,
        zoom_factor: float,
        zoom_center: Rigidbody,
        unit: astropy.units.Unit,
        vax: matplotlib.axes.Axes or None = None,
        plot_velocity: list[Rigidbody] = [],
    ):
        """
        Configures the axis limits for the current plot.
        """

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
        save_file: str,
        fade: int or None = None,
        log: bool = False,
        unit: astropy.units.Unit = u.meter,
        view_param: dict = dict(elev=20, azim=-45, roll=0),
        center_body: Rigidbody or None = None,
        center_view: bool = True,
        aspect_scale: tuple[float] = (1.0, 1.0, 1.0),
        zoom_factor: float = 1.0,
        zoom_center: Rigidbody or None = None,
        to_scale: bool = False,
        draw_acceleration: bool = False,
        plot_velocity: list[Rigidbody] = [],
        dpi: int or str = "figure",
        legend: str = "axes",
    ):
        """

        Creates an animation

        Parameters
        ----------

        steps_per_frame : int
        The amount of simulation steps that should be covered in one frame.

        framesep : int
        The seperation between each frame in milliseconds.
        The framerate (fps) is calculated by 1 / framesep.

        save_file : str
        The path of the output file. Available file types are: MP4 and GIF

        fade : int or None, optional
        The number of frames after which the trajectories will fade out.
        Default is `None`, meaning no fade effect.

        log : bool, optional
        Apply a symlog norm to all axes. This feature is mostly not needed and untested.
        Default is `False`.

        unit : astropy.units.Unit, optional
        The unit used for all lengths. Default is meter (`astropy.units.meter`).

        view_param : dict, optional
        The parameters to pass to the mpl_toolkits.mplot3d.axes3d.Axes3D.view_init
        method, determining the view point of the animation.

        center_body : Rigidbody or None, optional
        The body which should be the center of the reference frame.

        center_view : bool, optional
        Wether the axes limits should be set, so that they are static during
        the animation. This is done by setting the limits to the maximum and minimum
        values that are ever reached by any body. This option is recommended to
        get static an optically stable plot.
        Default is `True`.

        aspect_scale : tuple[float], optional
        The factor to multiply the aspect of the axes by. A value of `(1, 1, 1)`
        says, that the axes (x, y, z) should be scaled equally, which preserves
        the relative true scale of the axes.
        Default is equal scale, meaning `(1, 1, 1)`.

        zoom_factor : float, optional
        The factor by which the scale of the axes is enlarged.
        Default is `1`, meaning no zoom.

        zoom_center : Rigidbody or None, optional
        The center of the zoom. Default is `None`.

        to_scale : bool, optional
        Experimental feature, which increases the size of the markers of the planets.
        This is not really 'to scale', but increases or decreases the size of the
        markers according to
        the radius of the rigidbodies. If set, this disables the display of a legend.
        Default is `False`.

        draw_acceleration : bool, optional
        Wether to show arrows in the direction of the intrinsic acceleration.
        Default is `False`.

        plot_velocity : list[Rigidbody], optional
        A list of rigidbodies for which to show a plot of their relative orbital
        velocity. If this list is empty, there will be no plot of the orbital
        velocities.

        dpi : int or str, optional
        The dpi of the animation. Default is `figure`, meaning that the dpi is
        determined by the size of the figure

        legend : str, optional
        Determines the type of the legend. If set to `fig` the legend is
        drawn in the reference frame of the figure, if set to `axes` the
        legend is drawn onto the axes itself. Default is `axes`.

        Returns
        -------

        fig, ax : matplotlib.figure.Figure, mpl_toolkits.mplot3d.axes3d.Axes3D
        The figure and axes objects.

        """

        frames = self.n_it // steps_per_frame

        if len(plot_velocity) == 0:
            fig = plt.figure(figsize=(7, 7), layout="constrained")
            ax = fig.add_subplot(projection="3d")
            vax = None
        else:
            fig = plt.figure(figsize=(7, 7), layout="constrained")
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
                        *(current_acceleration * (self._get_max_length() * 1e-3)),
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
            legend=legend,
        )

        writer = None
        if save_file[-4:].lower() == "gif":
            writer = animation.PillowWriter(
                fps=1 / (framesep * 1e-3),
                bitrate=-1,
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
                ani.save(
                    save_file, writer=writer, progress_callback=progress_func, dpi=dpi
                )

        return fig, ax

    def plot_state(
        self,
        log: bool = False,
        unit: astropy.units.Unit = u.meter,
        view_param: dict = dict(elev=20, azim=-45, roll=0),
        center_body: Rigidbody or None = None,
        center_view: bool = False,
        aspect_scale: tuple[float] = (1.0, 1.0, 1.0),
        zoom_factor: float = 1.0,
        zoom_center: Rigidbody or None = None,
        save_file: str or None = None,
        to_scale: bool = False,
        legend: str = "axes",
    ):
        """
        Plots the current state of the simulation.

        Parameters
        ----------

        log : bool, optional
        Apply a symlog norm to all axes. This feature is mostly not needed and untested.
        Default is `False`.

        unit : astropy.units.Unit, optional
        The unit used for all lengths. Default is meter (`astropy.units.meter`).

        view_param : dict, optional
        The parameters to pass to the mpl_toolkits.mplot3d.axes3d.Axes3D.view_init
        method, determining the view point of the animation.

        center_body : Rigidbody or None, optional
        The body which should be the center of the reference frame.

        center_view : bool, optional
        Wether the axes limits should be set, so that they are static during
        the animation. This is done by setting the limits to the maximum and
        minimum values that are ever reached by any body. This option is recommended
        to get static an optically stable plot.
        Default is `True`.

        aspect_scale : tuple[float], optional
        The factor to multiply the aspect of the axes by. A value of `(1, 1, 1)`
        says, that the axes (x, y, z) should be scaled equally, which preserves
        the relative true scale of the axes.
        Default is equal scale, meaning `(1, 1, 1)`.

        zoom_factor : float, optional
        The factor by which the scale of the axes is enlarged. Default is `1`,
        meaning no zoom.

        zoom_center : Rigidbody or None, optional
        The center of the zoom. Default is `None`

        save_file : str or None, optional
        The path of the output file. Default is `None`, meaning no
        file will be saved.

        to_scale : bool, optional
        Experimental feature, which increases the size of the markers of the planets.
        This is not really 'to scale', but increases or decreases the size of the
        markers according to the radius of the rigidbodies. If set, this disables
        the display of a legend. Default is `False`.

        legend : str, optional
        Determines the type of the legend. If set to `fig` the legend is drawn
        in the reference frame of the figure, if set to `axes` the legend is
        drawn onto the axes itself. Default is `axes`.

        Returns
        -------

        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects.

        """

        fig = plt.figure(figsize=(7, 7), layout="constrained")
        ax = fig.add_subplot(projection="3d")

        fig.suptitle(
            f"Simulation time:\t{timedelta(seconds=int(self.time))}\t|"
            "\tStep:\t{self.n_it}"
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
            legend=legend,
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
