import matplotlib as mpl
import numpy as np
import toml
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time

MARKER_SIZE = mpl.rcParams["lines.markersize"]


class Rigidbody:
    def __init__(
        self,
        name,
        position,
        velocity,
        mass,
        radius,
        body_color,
        trail_color,
        exclude_bodies=[],
        body_alpha=1,
        trail_alpha=1,
        intrinsic_acceleration=np.zeros(3),
        marker="o",
        fix_marker_size=False,
    ):

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
        self.intrinsic_acceleration = np.array(intrinsic_acceleration)
        self.accelerations = np.array([])
        self.marker = marker
        self.fix_marker_size = fix_marker_size

        self._original_state = self.get_state()

    def _copy_state(self, state):
        state = state.copy()
        unique_types = [np.ndarray, list, tuple, dict]
        for key, value in state.items():
            if np.any([isinstance(value, utype) for utype in unique_types]):
                state[key] = value.copy()

        return state

    def get_state(self):
        return self._copy_state(self.__dict__)

    def get_original_state(self):
        return self._copy_state(self._original_state)

    @classmethod
    def copy(cls, body):
        cls = cls(*([None] * 7))
        cls.__dict__ = body.get_state()
        return cls

    def reset(self):
        original_state = self.get_original_state()
        self.__dict__ = original_state.copy()
        self.__dict__["_original_state"] = original_state.copy()

    def exclude_body(self, body, hard=True):
        self.exclude_bodies.append(body)

        if hard:
            self._original_state["exclude_bodies"].append(body)

    def __str__(self):
        return "\n".join("%s: %s" % item for item in self.get_state().items())

    def info(self):
        print(f"---- Info for {self.name} ----")
        print(str(self))

    def get_current_position(self):
        return self.positions[-3:]

    def get_current_velocity(self):
        return self.velocities[-3:]

    def get_abs_velocities(self, relative_to=None):
        if relative_to is not None:
            velocities = np.reshape(self.velocities - relative_to.velocities, (-1, 3))
        else:
            velocities = np.reshape(self.velocities, (-1, 3))
        return np.linalg.norm(velocities, axis=1).ravel()

    def get_min_pos(self, coordinate, center_body=None):
        if center_body is not None:
            return np.min(
                np.reshape(self.positions - center_body.positions, (-1, 3))[
                    :, coordinate
                ]
            )
        else:
            return np.min(np.reshape(self.positions, (-1, 3))[:, coordinate])

    def get_max_pos(self, coordinate, center_body=None):
        if center_body is not None:
            return np.max(
                np.reshape(self.positions - center_body.positions, (-1, 3))[
                    :, coordinate
                ]
            )
        else:
            return np.max(np.reshape(self.positions, (-1, 3))[:, coordinate])

    def get_marker_size(self, to_scale=False, max_length=0, unit=u.meter):
        if to_scale and not self.fix_marker_size:
            return 70000 * self.radius * u.meter / (max_length * unit)
        elif self.fix_marker_size:
            return MARKER_SIZE**2
        else:
            return MARKER_SIZE * np.pow(self.radius / R_earth.value, 0.2)

    def stop_acceleration(self):
        self.intrinsic_acceleration = np.zeros(3)

    def accelerate(self, acceleration, unit=u.meter / u.second**2):
        self.intrinsic_acceleration = (
            (acceleration * unit).to(u.meter / u.second**2).value
        )

    def move(self, time):
        self.accelerations = np.append(self.accelerations, self.intrinsic_acceleration)
        self.velocities = np.append(
            self.velocities,
            self.get_current_velocity()
            + (self.acceleration + self.accelerations[-3:]) * time,
        )
        self.positions = np.append(
            self.positions,
            self.get_current_position() + self.get_current_velocity() * time,
        )

    @classmethod
    def from_name(
        cls, name, time=Time.now(), marker="o", config="default_planets.toml"
    ):

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
