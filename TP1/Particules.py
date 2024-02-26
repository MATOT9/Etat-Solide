from functools import partial
from scipy import constants as cte
import numpy as np
import random
import vpython as vp
# Fichier contenant les fonctions partagées entre TDS_2Dsimulation.py et TDS_2Ddrude.py


class Particule:
    def __init__(self, position: vp.vector, qteMvt: vp.vector, masse: float, rayon: float, charge: float=0, couleur: vp.color=vp.color.gray(0.7)):
        self.pos: vp.vector = position
        self.p: vp.vector = qteMvt
        self.masse: float = masse
        self.rayon: float = rayon
        self.charge: float = charge
        self.sphere = vp.simple_sphere(pos=position, radius=rayon, color=couleur)


    def update(self, dt: float, L: float, E: float) -> None:
        # Le champ électrique accélère la particule
            # À faire

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
    @staticmethod
    def checkCollisionsAtomes(atomes: list[Particule]) -> list[list[int]]:
        hitlist: list[list[int]] = []

        # distance critique où les 2 sphères entre en contact à la limite de leur rayon
        r2: float = (2*atomes[0].rayon)**2

        for i in range(len(atomes)):
            posi: vp.vector = atomes[i].pos
            for j in range(i) :
                posj: vp.vector = atomes[j].pos

                # la boucle dans une boucle itère pour calculer cette distance
                # vectorielle dr entre chaque paire de sphère
                dr: vp.vector = posi - posj
                if vp.mag2(dr) < r2:
                    # liste numérotant toutes les paires de sphères en collision
                    hitlist.append([i,j])
        return hitlist


    @staticmethod
    def collisionAtomes(atome1: Particule, atome2: Particule, T: float) -> None:
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

        # Calcule la nouvelle quantité de mouvement de l'électron avec sa norme distribuée selon une distribution
        # Maxwell-Boltzman avec sa direction aléatoire
            # À implémenter

        # Calcule la distance à parcourir pour que l'électron quitte le noyau avec sa nouvelle trajectoire et avance l'électron
        # pour ne pas qu'il interagisse avec le même coeur plusieurs fois d'affilée
            # À implémenter


    @staticmethod
    def checkCollisionsCoeurs(electrons: list[Particule], coeurs: list[Particule]) -> list[list[int]]
        hitlist: list[int] = []

        # distance critique où les 2 sphères entre en contact à la limite de leur rayon
        r2: float = (atomes[0].rayon)**2

        for i in range(len(coeurs)) :
            coeur = coeurs[i]
            # Distance pour qu'il y ait une collision, on assume que tous les électrons on la même taille
            r: float = electrons[0].rayon+coeur.rayon

            for j in range(len(electrons)):
            electron = electrons[j]
                # La boucle dans une boucle itère pour calculer cette distance
                # vectorielle dr entre électron et coeur
                dr: vp.vector = electron.pos - coeur.pos
                if vp.mag(dr) < r:
                    # Liste numérotant tous les électrons et les coeurs en collision
                    hitlist.append([j, i])
        return hitlist


    @staticmethod
    def collisionsElectrons(electron: Particule, coeur: Particule) -> None:
        # Positions relative de l'électron et du coeur
        rrel: vp.vector = electron.pos - coeur.pos

        # Exclusion de cas où il n'y a pas de changements à faire, 2 cas:
        # exactly same velocities si et seulement si p de l'électron devient nul, pas de collision
        # one atom went all the way through another, la collision a été "manquée" à l'intérieur du pas deltax
        if vp.mag(electron.p) == 0 or rrel.mag > atome1.rayon+atome2.rayon:
            return None


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
