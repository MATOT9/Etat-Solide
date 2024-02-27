from __future__ import annotations
from copy import deepcopy
from functools import partial
from scipy import constants as cte
from scipy.stats import maxwell
import numpy as np
import random
import vpython as vp
# Fichier contenant les fonctions partagées entre TDS_2Dsimulation.py et TDS_2Ddrude.py


class Particule:
    def __init__(self, rayon: float, position: vp.vector, p: vp.vector = vp.vector(0,0,0),
                 masse: float = 0, charge: float = 0, couleur: vp.color = None):
        self.pos: vp.vector = position
        self.p: vp.vector = p
        self.masse: float = masse
        self.rayon: float = rayon
        self.charge: float = charge
        if couleur:
            self.sphere = vp.simple_sphere(pos=position, radius=rayon, color=couleur)


    def update(self, dt: float, L: float, E: float = 0) -> None:
        # Le champ électrique accélère la particule
        self.p += self.charge*E*vp.vector(1, 0, 0)

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
        if self.sphere:
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


    @staticmethod
    def checkCollisionsElectrons(electrons: list[Particule], coeurs: list[Particule]) -> list[list[int]]:
        hitlist: list[int] = []

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
    def collisionsElectrons(electron: Particule, coeur: Particule, T: float) -> None:
        # Positions relative de l'électron et du coeur
        rrel: vp.vector = electron.pos - coeur.pos
        r: float = vp.mag(rrel)
        R: float = electron.rayon+coeur.rayon

        # Exclusion de cas où il n'y a pas de changements à faire, 2 cas:
        # exactly same velocities si et seulement si p de l'électron devient nul, pas de collision
        # one atom went all the way through another, la collision a été "manquée" à l'intérieur du pas deltax
        if vp.mag(electron.p) == 0 or rrel.mag > R:
            return None

        # Calcule la nouvelle quantité de mouvement de l'électron avec sa norme distribuée
        # selon une distribution Maxwell-Boltzman avec sa direction aléatoire
        scale: float = np.sqrt(electron.masse*cte.k*T)
        p: float = maxwell.rvs(scale=scale)
        phi: float = 2 * np.pi * random.random()
        px: float = p*np.cos(phi)
        py: float = p*np.sin(phi)
        electron.p = vp.vector (px, py, 0)

        # Calcule la distance à parcourir pour que l'électron quitte le noyau avec sa nouvelle trajectoire
        # et avance l'électron pour ne pas qu'il interagisse avec le même coeur plusieurs fois d'affilée
        # cos de l'angle entre -rrel et p
        cosTheta: float = vp.dot(-rrel, electron.p)/(r*p)
        # Distance à parcourir
        d = r*cosTheta + np.sqrt(r**2*cosTheta**2 + R**2-r**2)
        # On avance en direction de p d'une distance d
        electron.pos += vp.vector(px, py, 0)*d/p


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


    def setupParticules(L: float, masse: float, rayon: float, nParticules: int, T: float,
            analyseSphere: int, rand: bool = True render: bool = True) -> list[Particule]:
    # POSITION ET QUANTITÉ DE MOUVEMENT INITIALE DES SPHÈRES
    # Liste qui contient les atomes
    atoms: list[Particule] = []

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
        if rand:
            phi: float = 2 * np.pi * random.random()
            px: float = pavg * np.cos(phi)
            py: float = pavg * np.sin(phi)
            pInit = vp.vector(px, py, 0)
        else:
            pInit = vp.vector(pavg, 0, 0)

        # On ajoute un atome à la liste des atomes, si render=False, l'attribut sphere de l'atome n'est pas initialisé
        particule = partial(Particule, rayon, posInit, p=pInit, masse=masse)
        if render:
            couleur: vp.color = vp.color.gray(0.7)
        else:
            couleur = None

        atoms.append(particule(couleur=couleur))

    if render:
        atoms[analyseSphere].sphere.radius *= 2
        atoms[analyseSphere].sphere.color = vp.color.magenta

    return atoms


def placerMotif(motif: list[list[float, vp.vector]], pos: vp.vector, L, render: bool = True) -> int:
    # Nombre de coeurs placés dans le motif
    coeursPlaces: int = 0
    coeurs: list[Particule] = []

    for coeur in motif:
        copie = deepcopy(coeur)
        copie[1] += pos
        # Ne place les coeurs que si ils sont dans la boite
        if abs(copie[1].x) < L/2 and abs(copie[1].y) < L/2:
            # Ne rend les sphères visibles que si render est vrai
            if render:
                couleur: vp.color = vp.color.red
            else:
                couleur = None

            coeurs.append(Particule(*copie, couleur=couleur))
            coeursPlaces += 1

    return coeurs, coeursPlaces


def setupCoeurs(maille: list[vp.vector], motif: list[list[float, vp.vector]], L: float, render: bool = True) -> list[Particule]:
    # Place un premier motif au centre de la boite, place
    # un coeur uniquement si le coeur est dans la boite
    coeurs: list[Particule] = []
    coeursMotif, places = placerMotif(motif, vp.vector(0, 0, 0), L, render)
    coeurs.extend(coeursMotif)
    
    rayon: int = 0
    coeursPlaces: int = 0
    while True:
        # Nombre de coeurs placés au cours de l'itération, quitte
        # la boucle si aucun n'est placé durant l'itération en cours
        coeursPlaces = 0
        # On rempli couche par couche en partant du centre
        rayon += 1

        for n1 in range(-rayon, rayon+1):
            centre: vp.vector = n1*maille[0] + rayon*maille[1]
            coeursMotif, places = placerMotif(motif, centre, L, render)
            coeurs.extend(coeursMotif)
            coeursPlaces += places

            centre: vp.vector = n1*maille[0] - rayon*maille[1]
            coeursMotif, places = placerMotif(motif, centre, L, render)
            coeurs.extend(coeursMotif)
            coeursPlaces += places

        for n2 in range(-rayon+1, rayon):
            centre: vp.vector = rayon*maille[0] + n2*maille[1]
            coeursMotif, places = placerMotif(motif, centre, L, render)
            coeurs.extend(coeursMotif)
            coeursPlaces += places

            centre: vp.vector = -rayon*maille[0] + n2*maille[1]
            coeursMotif, places = placerMotif(motif, centre, L, render)
            coeurs.extend(coeursMotif)
            coeursPlaces += places

        if coeursPlaces == 0:
            break

    return coeurs
