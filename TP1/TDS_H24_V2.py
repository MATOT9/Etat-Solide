"""Hard-sphere kinetic gas theory simulation using plotly."""


from dataclasses import dataclass, field, InitVar

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
            The time step between frames.
        elasticity
            The coefficient of restitution. A value of 1 means 100% energy
            conservation after a collision.
        mass
            The mass of the simulated particles.
        n_atoms
            The number of particles to use in the simulation.
        r_atom
            The particle size. Defaults to 5 to be visible on the animation.
        scale
            The reduction factor for velocity calculation. The velocity of each
            atom is first computed according to kinetic gas theory. The scale
            factor decrease velocity to keep the particles visible. Defaults to
            2500.
        T
            The initial temperature used to compute velocity.

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

    # Constants
    box_size: int = 10
    dt: int | float = 1
    elasticity: int | float = 1  # Coefficient of restitution
    mass: float = cte.value("alpha particle mass")
    n_atoms: int = 200
    r_atom: int | float = 5
    scale: int = 2500
    T: InitVar[int | float] = 300

    fig: go.Figure = field(default_factory=go.Figure)
    pos: np.ndarray[np.float64] = field(init=False)
    vel: np.ndarray[np.float64] = field(init=False)

    def __post_init__(self, T):
        self.pos = np.random.uniform(
            low=0,
            high=self.box_size,
            size=(self.n_atoms, 2),
        )

        # Average velocity according to kinetic gas theory.
        # Velocity is scaled down to be visible in the animation.
        vavg = np.sqrt(3 * cte.k * T / self.mass) / self.scale
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
        self.set_layout()
        nrange = range(self.n_atoms)
        self.fig.frames = tuple(map(self.init_frames, nrange))
        for n, frame in enumerate(self.fig.frames):
            # Update frames with animation function
            self.update_particles(n, frame)

        if anim is True:
            self.fig.show()

    def set_layout(self):
        box_size = self.box_size
        particles = go.Scatter(
            x=self.pos[:, 0],
            y=self.pos[:, 1],
            mode="markers",
            marker=dict(size=self.r_atom, color="red"),
        )
        self.fig.add_traces(particles)

        # Add rectangular container
        container = go.Scatter(
            x=[0, box_size, box_size, 0, 0],
            y=[0, 0, box_size, box_size, 0],
            mode="lines",
            line=dict(color="black", width=2),
        )
        self.fig.add_trace(container)

        # Set layout
        self.fig.update_layout(
            xaxis=dict(range=[-0.5, box_size + 0.5]),
            yaxis=dict(range=[-0.5, box_size + 0.5]),
            title="Hard Sphere Gas Animation",
            template="plotly_dark",
            showlegend=False,
            width=600,
            height=600,
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
                                    {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                            }
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            None,
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

    def collisions(self, i):
        for j in range(i + 1, self.n_atoms):
            rrel = self.pos[i] - self.pos[j]
            dist = np.linalg.norm(rrel)

            if dist != 0 and dist < 2 * self.r_atom:
                # Collision detected, update velocities
                vrel = self.vel[i] - self.vel[j]
                dot_product = np.dot(rrel, vrel)
                if dot_product < 0:
                    self.vel[i] -= dot_product / dist**2 * rrel
                    self.vel[j] += dot_product / dist**2 * rrel

    def update_particles(self, n, frame):
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
        self.collisions(n)

        frame.data[0].x = self.pos[:, 0]
        frame.data[0].y = self.pos[:, 1]
