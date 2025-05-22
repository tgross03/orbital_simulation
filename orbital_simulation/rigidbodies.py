from pathlib import Path

import astropy
import matplotlib as mpl
import numpy as np
import toml
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from numpy.typing import ArrayLike

MARKER_SIZE = mpl.rcParams["lines.markersize"]


class Rigidbody:
    def __init__(
        self,
        name: str,
        position: ArrayLike,
        velocity: ArrayLike,
        mass: float,
        radius: float,
        body_color: str,
        trail_color: str,
        exclude_bodies: list = [],
        body_alpha: float = 1,
        trail_alpha: float = 1,
        intrinsic_acceleration: list[float] = [0, 0, 0],
        marker="o",
        fix_marker_size=False,
    ):
        """
        Initializes a new rigidbody.

        Parameters
        ----------

        name : str
        The name of the rigidbody. Has to be unique in one simulation!

        position : array_like
        The initial position of the body in meter.
        Has to have a size of 3.

        velocity : array_like
        The initial velocity of the body in meter / second.
        Has to have a size of 3.

        mass : float
        The mass of the body in kilogram.

        radius : float
        The radius of the body in meter.

        body_color : str
        The color of the dot representing the body. This must either
        be a named matplotlib color or a hex color code.

        trail_color : str
        The color of the trajectory of the body. This must either
        be a named matplotlib color or a hex color code.

        exclude_bodies : list[Rigidbody], optional
        A list of bodies to ignore during the simulation. This means,
        that neither this body, nor the excluded body will experience
        any gravitational acceleration due to the other. This can e.g.
        be used to simulate the motion of a single space craft on different
        possible trajectories. In this case not excluding these bodies
        would result in extremely hight accelerations since the spacecrafts
        would likely start at the same position, experiencing infinite
        acceleration in the first frame.

        body_alpha : float, optional
        The alpha value for the dot representing the body. Default is `1`.

        trail_alpha : float, optional
        The alpha value for the trajectory of the body. Default is `1`.

        intrinsic_acceleration : array_like, optional
        The intrinsic_acceleration acceleration of the rigidbody. This
        describes all accelerations that are not caused due to the
        gravitational force. This could be due to the body propelling
        itself (spacecraft).
        Default is a null-vector.

        marker : str, optional
        The marker of the point representing the body in a plot.
        Has to be a valid matplotlib marker. Default is `o` (a filled dot).

        fix_marker_size : bool, optional
        Wether to scale the size of the marker if the scaling is activated
        in the plotting options of the simulation.

        """

        self.name = name
        self.acceleration = np.zeros(3)
        self.positions = np.array(position)
        self.velocities = np.array(velocity)
        self.mass = mass
        self.radius = radius
        self.body_color = body_color
        self.trail_color = trail_color
        self.exclude_bodies = exclude_bodies.copy()
        self.body_alpha = body_alpha
        self.trail_alpha = trail_alpha
        self.intrinsic_acceleration = np.array(intrinsic_acceleration.copy()).copy()
        self.accelerations = np.array([])
        self.marker = marker
        self.fix_marker_size = fix_marker_size

        self._original_state = self._copy_state(self.get_state())

    def _copy_state(self, state: dict):
        """
        Creates a deep copy of the given state.
        """
        state = state.copy()
        unique_types = [np.ndarray, list, tuple, dict]
        for key, value in state.items():
            if np.any([isinstance(value, utype) for utype in unique_types]):
                state[key] = value.copy()

        return state

    def get_state(self):
        """
        Get a dictionary representation of the current state
        of the rigidbody.

        Returns
        -------

        state : dict
        The current state of the rigidbody.

        """
        return self._copy_state(self.__dict__)

    def get_original_state(self):
        """
        Get the original state of the rigidbody.

        Returns
        -------

        original_state : dict
        The original state of the rigidbody.

        """
        return self._copy_state(self._original_state)

    @classmethod
    def copy(cls, body: "Rigidbody", name: str = None):
        """
        Creates a deep copy of the rigidbody.

        Parameters
        ----------

        body : Rigidbody
        The body to copy.

        name : str or None, optional
        The new name for the copied Rigidbody.
        Default is `None`, meaning the name should remain
        the same.

        """
        cls = cls(*([None] * 7))
        cls.__dict__ = body.get_state()

        if name is not None:
            cls.name = name
        return cls

    def reset(self):
        """
        Resets the rigidbody to its original state.
        """
        original_state = self.get_original_state()
        self.__dict__ = original_state.copy()
        self.__dict__["_original_state"] = original_state.copy()

    def exclude_body(self, body: "Rigidbody", hard: bool = True):
        """
        Excludes a rigidbody from being considered in gravity calculations.
        This means, that neither this body, nor the excluded body
        will experience any gravitational acceleration due to the other.
        This can e.g. be used to simulate the motion of a single space
        craft on different possible trajectories.
        In this case not excluding these bodies would result in
        extremely hight accelerations since the spacecrafts
        would likely start at the same position, experiencing infinite
        acceleration in the first frame.

        Parameters
        ----------

        body : Rigidbody
        The body to exclude.

        hard : bool, optional
        Wether to add the exclusion to the original state
        of the rigidbody. This means that the body will still
        be excluded after the rigidbody has been reset.

        """
        self.exclude_bodies.append(body)

        if hard:
            self._original_state["exclude_bodies"].append(body)

    def __str__(self):
        return "\n".join("%s: %s" % item for item in self.get_state().items())

    def info(self):
        """
        Prints the current state of the rigidbody and
        its properties.
        """
        print(f"------ Rigidbody: {self.name} ------")
        print(str(self))

    def get_current_position(self):
        """
        Get the current position of the rigidbody.

        Returns
        -------

        position : array_like
        The current position of the rigidbody.

        """
        return self.positions[-3:]

    def get_current_velocity(self):
        """
        Get the current velocity of the rigidbody.

        Returns
        -------

        position : array_like
        The current velocity of the rigidbody.

        """
        return self.velocities[-3:]

    def get_abs_velocities(self, relative_to: "Rigidbody" = None):
        """
        Get the absolute velocities relative to a given rigidbody.

        Parameters
        ----------

        relative_to : Rigidbody or None, optional
        The body the absolute velocities are supposed to be relative to.
        Default is `None`, meaning the velocities are supposed to be
        relative to the center of the coordinate system.

        Returns
        -------

        abs_velocities : array_like
        The absolute velocities relative to the given body.

        """
        if relative_to is not None:
            velocities = np.reshape(self.velocities - relative_to.velocities, (-1, 3))
        else:
            velocities = np.reshape(self.velocities, (-1, 3))
        return np.linalg.norm(velocities, axis=1).ravel()

    def get_min_pos(self, coordinate: int, center_body: "Rigidbody" = None):
        """
        Get the minimal value of a specific coordinate.

        Parameters
        ----------

        coordinate : int
        The index of the coordinate to return (x = 0, y = 1, z = 2).

        center_body : Rigidbody or None
        The body the coordinate should be relative to.

        Returns
        -------

        min_pos : float
        The minimal coordinate of the (relative) positions.

        """
        if center_body is not None:
            return np.min(
                np.reshape(self.positions - center_body.positions, (-1, 3))[
                    :, coordinate
                ]
            )
        else:
            return np.min(np.reshape(self.positions, (-1, 3))[:, coordinate])

    def get_max_pos(self, coordinate: int, center_body: "Rigidbody" = None):
        """
        Get the maximal value of a specific coordinate.

        Parameters
        ----------

        coordinate : int
        The index of the coordinate to return (x = 0, y = 1, z = 2).

        center_body : Rigidbody or None, optional
        The body the coordinate should be relative to.

        Returns
        -------

        min_pos : float
        The maximal coordinate of the (relative) positions.

        """
        if center_body is not None:
            return np.max(
                np.reshape(self.positions - center_body.positions, (-1, 3))[
                    :, coordinate
                ]
            )
        else:
            return np.max(np.reshape(self.positions, (-1, 3))[:, coordinate])

    def get_marker_size(
        self,
        to_scale: bool = False,
        max_length: float = 0,
        unit: astropy.units.Unit = u.meter,
    ):
        """
        Get the size of the marker. Experimental function, might not
        give the expected results.

        Parameters
        ----------

        to_scale : bool, optional
        Wether the size of the marker should be to scale to the radius.

        max_length : float, optional
        The maximum length of the plot.

        unit : astropy.units.Unit, optional
        The unit in which the maximum length is given.


        Returns
        -------

        marker_size : float
        The size of the marker.

        """
        if to_scale and not self.fix_marker_size:
            return 70000 * self.radius * u.meter / (max_length * unit)
        elif self.fix_marker_size:
            return MARKER_SIZE**2
        else:
            return MARKER_SIZE * np.pow(self.radius / R_earth.value, 0.2)

    def stop_acceleration(self):
        """
        Stops the current intrinsic acceleration.
        """
        self.intrinsic_acceleration = np.zeros(3)

    def accelerate(
        self,
        acceleration: float or ArrayLike,
        mode: str = "vector",
        unit: astropy.units.Unit = u.meter / u.second**2,
        relative_to: "Rigidbody" = None,
    ):
        """
        Adds an acceleration to the intrinsic acceleration.

        Parameters
        ----------

        acceleration : float or array_like
        The acceleration to add.

        mode : str, optional
        The mode of the acceleration. If this is set to `vector`,
        the provided the acceleration has to be an array_like object.
        Other modes `prograde`, `retrograde`, `radial_in` and `radial_out`
        requires a scalar value as the acceleration.

        unit : astropy.units.Unit, optional
        The unit of the provided acceleration. Default is meter / second^2.

        relative_to : Rigidbody or None, optional
        The body, the acceleration should be relative to.
        Defaul is `None`, meaning the acceleration is relative to the center
        of the coordinate system.

        """
        if relative_to is None:
            v_rel = np.zeros(3)
        else:
            v_rel = relative_to.get_current_velocity()

        acceleration = (acceleration * unit).to(u.meter / u.second**2).value

        match mode:
            case "vector":
                self.intrinsic_acceleration = self.intrinsic_acceleration + acceleration
            case "prograde":
                if not np.isscalar(acceleration):
                    raise ValueError(
                        "If mode is not set to 'vector', "
                        "the acceleration has to be scalar"
                    )

                self.intrinsic_acceleration = (
                    self.intrinsic_acceleration
                    + acceleration
                    * (self.get_current_velocity() - v_rel)
                    / np.linalg.norm(self.get_current_velocity() - v_rel)
                )

            case "retrograde":
                if not np.isscalar(acceleration):
                    raise ValueError(
                        "If mode is not set to 'vector', "
                        "the acceleration has to be scalar"
                    )

                self.intrinsic_acceleration = (
                    self.intrinsic_acceleration
                    + -acceleration
                    * (self.get_current_velocity() - v_rel)
                    / np.linalg.norm(self.get_current_velocity() - v_rel)
                )

            case "radial_in":
                if not np.isscalar(acceleration):
                    raise ValueError(
                        "If mode is not set to 'vector', "
                        "the acceleration has to be scalar"
                    )

                self.intrinsic_acceleration = (
                    self.intrinsic_acceleration
                    + -acceleration
                    * (self.get_current_position() - relative_to.get_current_position())
                    / np.linalg.norm(
                        self.get_current_position() - relative_to.get_current_position()
                    )
                )

            case "radial_out":
                if not np.isscalar(acceleration):
                    raise ValueError(
                        "If mode is not set to 'vector', "
                        "the acceleration has to be scalar"
                    )

                self.intrinsic_acceleration = (
                    self.intrinsic_acceleration
                    + acceleration
                    * (self.get_current_position() - relative_to.get_current_position())
                    / np.linalg.norm(
                        self.get_current_position() - relative_to.get_current_position()
                    )
                )

    def move(self, dt: float):
        """
        Move the rigidbody according to the current acceleration.

        Parameters
        ----------

        dt : float
        The time the moving should take. In this time frame the movement
        is assumed to be linear.

        """
        self.accelerations = np.append(self.accelerations, self.intrinsic_acceleration)
        self.velocities = np.append(
            self.velocities,
            self.get_current_velocity()
            + (self.acceleration + self.accelerations[-3:]) * dt,
        )
        self.positions = np.append(
            self.positions,
            self.get_current_position() + self.get_current_velocity() * dt,
        )

    @classmethod
    def from_name(
        cls,
        name: str,
        time: astropy.time.Time = Time.now(),
        marker: str = "o",
        config: str = None,
    ):
        """
        Gets a real rigidbody from a name given in a configuration file.

        Parameters
        ----------

        name : str
        The name of the rigidbody.

        time : astropy.time.Time
        The moment time at which the position and velocity of the body
        should be fetched.
        Default is the current time.

        marker : str, optional
        The marker that should be used to display the rigidbody in a plot.
        Has to be a valid matplotlib marker. Default is `o` (a filled dot).

        config : str or None, optional
        The path to the config file containing the rigidbody properties.
        Default is `None`, meaning the default configuration of the
        package is being used.

        """
        if config is None:
            config = str(Path(__file__).with_name("default_planets.toml"))

        sun_pos, sun_vel = get_body_barycentric_posvel(body="sun", time=time)
        pos, vel = get_body_barycentric_posvel(body=name, time=time)
        parameters = toml.load(config)

        if name not in parameters:
            raise KeyError("This celestial body is not in the given config.")

        parameters = parameters[name]

        cls = Rigidbody(
            name=name[0].upper() + name[1:],
            position=(pos.xyz - sun_pos.xyz).to(u.meter).value,
            velocity=(vel.xyz - sun_vel.xyz).to(u.meter / u.second).value,
            mass=parameters["mass"],
            radius=parameters["equatorial_radius"],
            body_color=parameters["body_color"],
            trail_color=parameters["trail_color"],
            marker=marker,
        )

        return cls
