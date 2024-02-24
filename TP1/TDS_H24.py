"""Hard-sphere gas simulation using vpython."""

__author__ = "Bruce Sherwood", "Maxime Tousignant-Tremblay"


from typing import Iterable
from functools import partial
from multiprocessing import Process
from dataclasses import dataclass, field, InitVar

import numpy as np
import vpython as vp
from scipy import constants as cte


GRAY = vp.color.gray(0.7)


@dataclass(slots=True, repr=False, order=False)
class HardSphere:
    """Hard-sphere gas simulation object."""

    dt: float = 1e-5
    L: int = 1
    mass: float = cte.value("alpha particle mass")
    Natoms: int = 200
    Ratom: float = 0.01
    T: InitVar[int | float] = 300

    anim: vp.canvas = field(init=False)
    apos: Iterable | tuple = field(init=False)
    Atoms: Iterable | list = field(init=False)
    d: float = field(init=False)
    nrange: list = field(init=False)
    p: Iterable | tuple = field(init=False)
    pos: np.ndarray = field(init=False)
    pavg: float = field(init=False)


    def __post_init__(self, T):
        self.d = self.L/2 + self.Ratom
        self.nrange = range(self.Natoms)

        # Variables that are not used until runing the simulation.
        self.anim = None
        self.Atoms = None

        # Thermodynamic energy array
        self.pavg = (2 * self.mass * 1.5 * cte.k * T)**0.5

        # Initial momentum of the spheres 
        self.p = self.get_p()

        # List of spheres initial position. Origin is the center of the box.
        self.pos = self.L * (np.random.rand(self.Natoms, 3) - 0.5)
        self.pos[:, -1] = 0
        self.apos = map(vp.vec, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        self.apos = list(self.apos)

    def get_p(self):
        """Generate initial p values of the spheres.

        Returns
        -------
            p
                The list of initial momentum of all spheres.

        """
        phi = 2 * np.pi * np.random.rand(self.Natoms)
        px = self.pavg * np.cos(phi)
        py = self.pavg * np.sin(phi)
        pz = np.zeros(self.Natoms)
        p = list(map(vp.vec, px, py, pz))
        return p

    def create_spheres(self, idx, pos):
        """Generate spheres to represent particles in the simulation.

        Parameters
        ----------
            idx
                The atom's index number.
            pos
                The 3D position array of the atom.

        Returns
        -------
            sphere
                A vpython sphere object.

        """
        simple_sphere = partial(vp.simple_sphere, pos=vp.vec(*pos))
        if idx == 0:
            # keeps a bigger and colored sphere among all the gray ones.
            sphere = simple_sphere(radius=0.03, color=vp.color.magenta)
            return sphere
        else:
            sphere = simple_sphere(radius=self.Ratom, color=GRAY)
            return sphere

    def checkCollisions(self, idx):
        """Identify collisions.
        
        A collision happens when the distance between the centers of two
        spheres is on the limit of interpenetrating.

        Parameter
        ---------
            idx
                The atom's index number.

        Returns
        -------
            n_dr
                The square of the norm of the intersphere distance dr.

        """
        # Critical distance where the 2 spheres come into contact.
        r2 = (2 * self.Ratom)**2
        ai = self.apos[idx]
        for j in range(idx):
            aj = self.apos[j]

            # Vector distance between each pair of sphere.
            dr = ai - aj

            if vp.mag2(dr) < r2:
                n_dr = [idx, j]
                return n_dr
        return

    def run(self, n: int):
        """Wrapper function to launch the simulation using instance method.
        
        To avoid vpython's infinite runing bug with jupyterlab, this function
        spawns an independent process to run the animation. This allows the
        script to continue it's execution even if a bug within vpython creates
        an infinite while loop that can only be stoped by manually closing the
        browser window. Communication between the process and the main script
        remains to be implemented.

        Parameter
        ---------
            n
                The number of iteration to use in the animation's main loop.

        """
        proc = Process(target=self._run, args=(n,))
        proc.start()

    def _run(self, n: int):
        """Function to launch the simulation.

        Parameter
        ---------
            n
                The number of iteration to use in the animation's main loop.

        """
        # Creating local copies of instance variables.
        d = self.d
        L = self.L
        dt = self.dt
        mass = self.mass
        nrange = self.nrange
        Ratom = self.Ratom

        # Animation preparation
        self.anim = vp.canvas(width=750, height=500)
        self.anim.range = L
        self.anim.title = 'Cinétique des gaz parfaits'
        vecs = [
                vp.vec(-d, -d, 0),
                vp.vec(d, -d, 0),
                vp.vec(d, d, 0),
                vp.vec(-d, d, 0),
                vp.vec(-d, -d, 0),
            ]
        vp.curve(
            pos=vecs,
            color=GRAY,
            radius=0.005,
        )
        self.Atoms = tuple(map(self.create_spheres, self.nrange, self.pos))
        self.apos = map(vp.vec, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        self.apos = list(self.apos)

        for _ in range(n):
            # Limit simulation speed to stay visible.
            vp.rate(300)

            # Moves all spheres with one spatial step deltax.
            vitesse = list(map(lambda pi: pi / mass, self.p))
            deltax = list(map(lambda vi: vi * dt, vitesse))
            for i in nrange:
                # Atom's new position after dt time step.
                self.Atoms[i].pos = self.apos[i] = self.apos[i] + deltax[i]

                # Preserves momentum upon collisions with box walls.
                loc = self.apos[i]
                if abs(loc.x) > L/2 and loc.x < 0:
                    # Flip x component to the left wall.
                    self.p[i].x = abs(self.p[i].x)
                elif abs(loc.x) > L/2:
                    # Flip x component to the right wall.
                    self.p[i].x = -abs(self.p[i].x)

                if abs(loc.y) > L/2 and loc.y < 0:
                    # Flip y component at the bottom wall.
                    self.p[i].y = abs(self.p[i].y)
                elif abs(loc.y) > L/2:
                    # Flip y component at the top wall.
                    self.p[i].y = -abs(self.p[i].y)

            # Find collisions.
            hitlist = map(self.checkCollisions, nrange)
            hitlist = filter(lambda x: x is not None, hitlist)

            # Preserves momentum in collisions between spheres.
            for ij in hitlist:
                # Define new variables for each pair of colliding spheres 
                i, j = ij
                ptot = self.p[i] + self.p[j]

                # Speed of the barycentric/center-of-momentum reference frame.
                Vcom = ptot / (2 * mass)
                posi = self.apos[i]
                posj = self.apos[j]
                vi = self.p[i] / mass
                vj = self.p[j] / mass

                # Vectors for the distance between the centers of the 2 spheres
                # and the difference in speed between the 2 spheres.
                rrel = posi - posj
                vrel = vj - vi

                # Exclusion of cases where there are no changes to be made.
                if vrel.mag2 == 0 or rrel.mag > Ratom:
                    continue

                # Calculates the distance and time of interpenetration of hard
                # spheres which should not occur in this model.
                dx = vp.dot(rrel, vrel.hat)
                dy = vp.cross(rrel, vrel.hat).mag

                # Alpha is the angle of the triangle composed of rrel, path of
                # atom j, and a line from the center of atom i to the center
                # of atom j where atome j hits atom i.
                alpha = vp.asin(dy / (2 * Ratom))

                # Distance traveled into the atom from first contact.
                d = (2 * Ratom) * vp.cos(alpha) - dx

                # Time spent moving from first contact to position inside atom.
                deltat = d / vrel.mag

                # Changes interpenetration of spheres through collision kinetics.
                posi = posi - vi * deltat
                posj = posj - vj * deltat
                pcomi = self.p[i] - mass * Vcom
                pcomj = self.p[j] - mass * Vcom
                pcomi = pcomi - 2 * vp.dot(pcomi, rrel) * rrel.hat
                pcomj = pcomj - 2 * vp.dot(pcomj, rrel) * rrel.hat
                self.p[i] = pcomi + mass * Vcom
                self.p[j] = pcomj + mass * Vcom

                # Move forward deltat in time, bringing back to the same time
                # where the other spheres are in the iteration.
                self.apos[i] = posi + (self.p[i] / mass) * deltat
                self.apos[j] = posj + (self.p[j] / mass) * deltat


if __name__ == "__main__":
    # Legacy launcher to run the simulation by executing the script.
    # Using this backend loads the 'hs' object in the host's global
    # variables.
    hs = HardSphere()
    proc = Process(target=hs._run, args=(200,))
    proc.start()
