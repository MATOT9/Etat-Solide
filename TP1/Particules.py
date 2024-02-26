from functools import partial
from scipy import constants as cte
import numpy as np
import random
import vpython as vp
# Fichier contenant les fonctions partagées entre TDS_2Dsimulation.py et TDS_2Ddrude.py


class Particule:
    def __init__(self, position: vp.vector, qteMvt: vp.vector, masse: float, rayon: float, couleur: vp.color):
        self.pos: vp.vector = position
        self.p: vp.vector = qteMvt
        self.masse: float = masse
        self.rayon: float = rayon
        self.sphere = vp.simple_sphere(pos=position, radius=rayon, color=couleur)


    def update(self, dt: float, L: float) -> None:
        # Fait un pas à la particule pour un temps dt
        vitesse: vp.vector = self.p/self.masse
        self.pos += vitesse * dt

        # Fait rebondir la particule lors d'une collision avec la boite
        if self.pos.x < -L/2:
            self.p.x = abs(self.p.x)
        if self.pos.x > L/2:
            self.p.x = -abs(self.p.x)
        if self.pos.y < -L/2:
            self.p.y = abs(self.p.y)
        if self.pos.y > L/2:
            self.p.y = -abs(self.p.y)

        # Update la position des sphères pour l'animation
        self.sphere.pos = self.pos


# FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
def checkCollisions(particules: list[Particule]) -> list[list[int]]:
    hitlist: list[list[int]] = []

    # distance critique où les 2 sphères entre en contact à la limite de leur rayon
    r2: float = (2*particules[0].rayon)**2

    for i in range(len(particules)):
        posi: vp.vector = particules[i].pos
        for j in range(i) :
            posj: vp.vector = particules[j].pos

            # la boucle dans une boucle itère pour calculer cette distance
            # vectorielle dr entre chaque paire de sphère
            dr: vp.vector = posi - posj
            if vp.mag2(dr) < r2:
                # liste numérotant toutes les paires de sphères en collision
                hitlist.append([i,j])
    return hitlist


def collisionAtomes(atome1: Particule, atome2: Particule) -> None:
    # Positions et vitesses relatives des deux particules
    v1: vp.vector = atome1.p/atome1.masse
    v2: vp.vector = atome2.p/atome2.masse
    rrel: vp.vector = atome2.pos - atome1.pos
    vrel: vp.vector = v1 - v2

    # Exclusion de cas où il n'y a pas de changements à faire, 2 cas:
    # exactly same velocities si et seulement si le vecteur vrel devient nul, la trajectoire des 2 sphères continue alors côte à côte
    # one atom went all the way through another, la collision a été "manquée" à l'intérieur du pas deltax
    if vrel.mag2 == 0 or rrel.mag > atome1.rayon+atome2.rayon:
        return None

    # Calcule la distance et temps d'interpénétration des sphères dures qui ne doit pas se produire dans ce modèle
    dx: float = vp.dot(rrel, vrel.hat)
    dy: float = vp.cross(rrel, vrel.hat).mag
    # Alpha est l'angle du triangle composé de rrel, la trajectoire de l'atome 2 et la ligne
    # entre le centre de l'atome 1 vers le centre de l'atome 2 où les deux atomes se touchent
    alpha = vp.asin(dy / (atome1.rayon+atome2.rayon))
    # Distance parcourue à l'intérieur du l'atome à partir du premier contact
    d: float = (atome1.rayon+atome2.rayon) * vp.cos(alpha) - dx
    # Temps écoulé pour se déplacer de la première collisions à la position à l'intérieur de l'atome
    deltat: float = d / vrel.mag

    # Transform momenta to center-of-momentum (com) frame
    Vcom: vp.vector = (atome1.p+atome2.p) / (atome1.masse+atome2.masse)
    pcom1 = atome1.p - atome1.masse*Vcom
    pcom2 = atome2.p - atome2.masse*Vcom

    # Bounce in center-of-momentum (com) frame
    rrel = vp.hat(rrel)
    atome1.p -= 2 * vp.dot(pcom1, rrel) * rrel
    atome2.p -= 2 * vp.dot(pcom2, rrel) * rrel

    # Change l'interpénétration des sphères par la cinétique de collision,
    # puis avance de deltat dans le temps, ramenant au même temps où sont
    # rendues les autres sphères dans l'itération
    atome1.pos += (atome1.p / atome1.masse - v1) * deltat
    atome2.pos += (atome2.p / atome2.masse - v2) * deltat


def canvas(L: float, rParticule: float) -> None:
    gray = vp.color.gray(0.7)   # color of edges of container and spheres below
    animation = vp.canvas(width=750, height=500)
    animation.range = L
    animation.title = 'Cinétique des gaz parfaits'

    d: float = L/2 + rParticule
    cadre = vp.curve(color=gray, radius=0.005)
    cadre.append(
        [
            vp.vec(-d, -d, 0),
            vp.vec(d, -d, 0),
            vp.vec(d, d, 0),
            vp.vec(-d, d, 0),
            vp.vec(-d, -d, 0),
        ]
    )


def setupParticules(L: float, masse: float, rayon: float, nParticules: int, T: float, analyseSphere: int) -> list[Particule]:
    # POSITION ET QUANTITÉ DE MOUVEMENT INITIALE DES SPHÈRES
    # Liste qui contient les atomes
    atoms: list[vp.simple_sphere] = []

    # Principe de l'équipartition de l'énergie en thermodynamique statistique classique
    # La bonne formule n'a pas de 1.5 vu qu'on est en 2D???
    # pavg: float = (2 * masse* 1.5 * cte.k * T)**0.5 anciennement
    pavg: float = (2 * masse * cte.k * T)**0.5

    for i in range(nParticules):
        # Position aléatoire qui tient compte que l'origine est au centre de la boîte
        xyz: np.ndarray[float] = L * np.random.rand(3) - L/2
        xyz[-1]: float = 0
        posInit = vp.vector (*xyz)

        # Qte de mouvement initiale de direction aléatoire avec norme selon l'équipartition
        phi: float = 2 * np.pi * random.random()
        px: float = pavg * np.cos(phi)
        py: float = pavg * np.sin(phi)
        pInit = vp.vector (px, py, 0)

        # On ajoute un atome à la liste des atomes
        particule = partial(Particule, posInit, pInit, masse, rayon)
        if i == analyseSphere:
            # Garde une sphère plus grosse et colorée parmis toutes les grises
            atoms.append(particule(vp.color.magenta))
            atoms[-1].sphere.radius *= 2
        else: 
            atoms.append(particule(vp.color.gray(0.7)))

    return atoms
