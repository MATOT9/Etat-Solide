"""Hard-sphere kinetic gas theory simulation using plotly."""

__author__ = "Bruce Sherwood", "Maxime Tousignant-Tremblay"

# Standard library
from dataclasses import dataclass, field, InitVar

# Third party libraries
import numpy as np
from scipy import constants as cte
from plotly import graph_objs as go


@dataclass(slots=True, repr=False, order=False, eq=False)
class HardSphere:
    """Hard-sphere gas object.
    
    Parameters
    ----------
        box_size
            The size of the simulated box.
        dt
            The time step between frames. The velocity of each atom is first
            computed according to kinetic gas theory. Use this parameter to
            adjust overall velocity of particles to keep their motion visible
            on the animation. Defaults to 30 μs.
        elasticity
            The coefficient of restitution. A value of 1 means 100% energy
            conservation after a collision.
        mass
            The mass of the simulated particles.
        n_atoms
            The number of particles to use in the simulation. Defaults to 100.
        n_frames
            The number of frames in the animation. Affects duration, defaults
            to 100 frames.
        pix_size
            The size of the box in pixels. Defaults to 600 pixels.
        r_atom
            The particle size. Defaults to 0.005 to be visible.
        T
            The initial temperature used to compute velocity. Defaults to room
            temperature, 300 K.

    Attributes
    ----------
        fig
            The Plotly figure object of the animation.
        pos
            The position array of all atoms.
        vel
            The velocity array of all atoms.
        p
            The momentum array of all atoms.
        p2
            The dot product <p, p> array of all atoms.

    """

    # Class constants
    box_size: int = 1
    dt: int | float = 30e-6
    elasticity: int | float = 1  # Coefficient of restitution
    mass: float = cte.value("alpha particle mass")
    n_atoms: int = 100
    n_frames: int = 100
    pix_size: int = 600
    r_atom: int | float = 0.005
    T: InitVar[int | float] = 300

    # Class properties
    fig: go.Figure = field(default_factory=go.Figure)
    pos: np.ndarray[np.float64] = field(init=False)
    vel: np.ndarray[np.float64] = field(init=False)

    def __post_init__(self, T):
        # Generate initial positions with a uniform distribution
        self.pos = np.random.uniform(
            low=0,
            high=self.box_size,
            size=(self.n_atoms, 2),
        )

        # Average velocity according to kinetic gas theory
        vavg = np.sqrt(3 * cte.k * T / self.mass)
        self.vel = np.random.normal(loc=0, scale=vavg, size=(self.n_atoms, 2))

    @property
    def p(self):
        return self.vel * self.mass

    @property
    def p2(self, n=None):
        """Returns the momentum norm squared.
        
        Parameter
        ---------
            n
                The index of the particle to extract p squared.
                
        Returns
        -------
            _p2
                The dot product <p, p> of all particles or the selected n atom.

        """
        _p2 = np.linalg.norm(self.p, axis=1)**2
        if n is None:
            return _p2
        return _p2[n]

    def run(self, anim=True):
        if anim is True:
            self.set_layout()
            nrange = range(self.n_atoms)
            self.fig.frames = tuple(map(self.init_frames, nrange))
            for frame in self.fig.frames:
                # Update frames with animation
                self.update_particles(frame)

            self.fig.show()
        elif anim is False:
            for frame in range(self.n_frames):
                # Update frames without animation
                self.update_particles(frame, anim)
        else:
            raise TypeError("anim parameter must be a boolean")

    def set_layout(self):
        # Creating local copies of instance variables for faster access
        box_size = self.box_size
        pix_size = self.pix_size

        # Marker size (diameter) in Plotly is defined in pixels, not with
        # respect to the coordinates system. We need to make sure both of them
        # are proportional to get an accurate animation.
        marker_size = 2 * self.r_atom * self.pix_size / self.box_size

        particles = go.Scatter(
            x=self.pos[:, 0],
            y=self.pos[:, 1],
            mode="markers",
            name="",
            marker=dict(size=marker_size, color="red"),
        )
        self.fig.add_traces(particles)

        # Add rectangular container
        container = go.Scatter(
            x=[0, box_size, box_size, 0, 0],
            y=[0, 0, box_size, box_size, 0],
            mode="lines",
            line=dict(color="black", width=0.05),
        )
        self.fig.add_trace(container)

        # Set layout
        self.fig.update_layout(
            xaxis=dict(range=[0, box_size]),
            yaxis=dict(range=[0, box_size]),
            title="Hard Sphere Gas Animation",
            template="plotly_dark",
            showlegend=False,
            width=pix_size,
            height=pix_size,
        )

        # Set up animation
        self.fig.layout.updatemenus = [
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame":
                                    {"duration": self.n_frames, "redraw": True},
                                    "fromcurrent": True,
                            }
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            (None,),
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            }
                        ],
                        "label": "Pause",
                        "method": "animate",
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]

    def init_frames(self, idx):
        frame = go.Frame(
            data=[go.Scatter(x=[], y=[])],
            name=f"frame_{idx + 1}",
            layout=self.fig.layout,
        )
        frame.update(
            layout=self.fig.layout,
            data=self.fig.data,
            name=f"frame_{idx + 1}",
            traces=[0, 1],
        )
        return frame

    def handle_collisions(self):
        # Find relative positions and velocities for all pairs of particles
        rrel = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        dist = np.linalg.norm(rrel, axis=-1)

        # Exclude self-collisions and check against particle size
        mask = (dist > 0) & (dist <= 2 * self.r_atom)

        # Calculate relative velocity.
        vrel = self.vel[:, np.newaxis, :] - self.vel[np.newaxis, :, :]
        dot_product = np.sum(rrel * vrel, axis=-1)

        # Select only pairs with approaching particles (dot_product < 0)
        mask &= dot_product < 0

        # Update velocities using the collision formula for selected pairs
        rrel_selected = np.where(mask[..., np.newaxis], rrel, 0)
        dist_selected = np.where(mask, dist, np.inf)
        dv = (dot_product / dist_selected**2)[..., np.newaxis] * rrel_selected
        self.vel += np.sum(dv, axis=1)

    def update_particles(self, frame, anim=True):
        box_size = self.box_size

        # Update positions
        self.pos += self.vel * self.dt

        # Handle collisions with walls
        self.pos = np.where(self.pos < 0, 0, self.pos)
        self.pos = np.where(self.pos > box_size, box_size, self.pos)

        # Reflect velocities upon wall collision
        self.vel[self.pos == 0] *= -self.elasticity
        self.vel[self.pos == box_size] *= -self.elasticity

        # Handle collisions between particles
        self.handle_collisions()

        # Update frame data if working with animation
        if anim is True:
            frame.data[0].x = self.pos[:, 0]
            frame.data[0].y = self.pos[:, 1]
