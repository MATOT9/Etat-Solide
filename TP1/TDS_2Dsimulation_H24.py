"""
Created on Fri Feb  5 15:40:27 2021

#GlowScript 3.0 VPython

# Hard-sphere gas.

# Bruce Sherwood
# Claudine Allen
"""

# Standard library
from functools import partial
import random

# Third-party libraries
from scipy import constants as cte
import Particules
import numpy as np
import vpython as vp


# FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
def checkCollisions(particules: list[Particule]):
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


def main():
    # win = 500 # peut aider à définir la taille d'un autre objet visuel comme un histogramme proportionnellement à la taille du canevas.
    # Déclaration de variables influençant le temps d'exécution de la simulation
    dt: float = 1e-5       # pas d'incrémentation temporel
    timeLoopLen: int = 1000  # temps de simulation
    nAtoms: int = 200    # change this to have more or fewer atoms

    # Variables reliées à l'atome à étudier
    analyseSphere: int = 106  # à changer si analyse autre atome He (int [0, 199])
    # Listes des variables suivies entre les collisions
    dVec: list[vp.vector] = [vp.vector(0, 0, 0)]
    dScalaire: list[float] = [0]
    tCollision: list[float] = [0]

    # Déclaration de variables physiques "Typical values"
    mass: float = cte.value("alpha particle mass")
    rAtom: float = 0.01
    T: float = 300

    # CANEVAS DE FOND
    L: float = 1                # container is a cube L on a side
    gray = vp.color.gray(0.7)   # color of edges of container and spheres below
    animation = vp.canvas(width=750, height=500)
    animation.range = L
    animation.title = 'Cinétique des gaz parfaits'

    # ARÊTES DE BOÎTE 2D
    d: float = L/2 + rAtom
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

    # POSITION ET QUANTITÉ DE MOUVEMENT INITIALE DES SPHÈRES
    # Liste qui contient lies atomes
    atoms: list[vp.simple_sphere] = []

    # Principe de l'équipartition de l'énergie en thermodynamique statistique classique
    # La bonne formule n'a pas de 1.5 vu qu'on est en 2D???
    # pavg: float = (2 * mass* 1.5 * cte.k * T)**0.5 anciennement
    pavg: float = (2 * mass * cte.k * T)**0.5

    for i in range(nAtoms):
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
        particule = partial(Particule, posInit, pInit, mass, rAtom)
        if i == analyseSphere:
            # Garde une sphère plus grosse et colorée parmis toutes les grises
            lastPos: vp.vector = posInit
            atoms.append(particule(magenta))
        else: 
            atoms.append(particule(gray))

    # BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt
    for _ in range(timeLoopLen):
        # limite la vitesse de calcul de la simulation pour que l'animation soit
        # visible à l'oeil humain!
        vp.rate(300)

        [atom.update(dt, L) for atom in atoms]

        # LET'S FIND THESE COLLISIONS!!!
        hitlist: list[list[int]] = checkCollisions()

        # CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS ENTRE SPHÈRES
        for ij in hitlist:
            # nouvelles variables pour chaque paire de sphères encollision
            i, j = ij
            ptot = p[i] + p[j]
            posi = apos[i]
            posj = apos[j]
            vi = p[i] / mass
            vj = p[j] / mass

            # vitesse du référentiel barycentrique/center-of-momentum (com) frame
            Vcom = ptot / (2 * mass)

            # vecteurs pour la distance entre les centres des 2 sphères et pour la
            # différence de vitesse entre les 2 sphères
            rrel = posi - posj
            vrel = vj - vi

            # exclusion de cas où il n'y a pas de changements à faire, 2 cas:
            # exactly same velocities si et seulement si le vecteur vrel devient nul, la trajectoire des 2 sphères continue alors côte à côte
            # one atom went all the way through another, la collision a été "manquée" à l'intérieur du pas deltax
            if vrel.mag2 == 0 or rrel.mag > 2*rAtom:
                continue

            # calcule la distance et temps d'interpénétration des sphères dures qui ne doit pas se produire dans ce modèle
            dx = vp.dot(rrel, vrel.hat)
            dy = vp.cross(rrel, vrel.hat).mag
            alpha = vp.asin(dy / (2 * rAtom))  # alpha is the angle of the triangle composed of rrel, path of atom j, and a line from the center of atom i to the center of atom j where atome j hits atom i
            d = (2 * rAtom) * vp.cos(alpha) - dx # distance traveled into the atom from first contact
            deltat = d / vrel.mag         # time spent moving from first contact to position inside atom

            # CHANGE L'INTERPÉNÉTRATION DES SPHÈRES PAR LA CINÉTIQUE DE COLLISION
            posi = posi-vi*deltat
            posj = posj-vj*deltat

            # transform momenta to center-of-momentum (com) frame
            pcomi = p[i]-mass*Vcom
            pcomj = p[j]-mass*Vcom

            # bounce in center-of-momentum (com) frame
            rrel = vp.hat(rrel)
            pcomi -= 2 * vp.dot(pcomi, rrel) * rrel
            pcomj -= 2 * vp.dot(pcomj, rrel) * rrel

            # transform momenta back to lab frame
            p[i] = pcomi + mass * Vcom
            p[j] = pcomj + mass * Vcom

            # move forward deltat in time, ramenant au même temps où sont rendues
            # les autres sphères dans l'itération
            apos[i] = posi + (p[i] / mass) * deltat
            apos[j] = posj + (p[j] / mass) * deltat

        # Met à jour la liste des variables à suivre
        # Unpack les paires de collisions
        collisions: list[int] = []
        [collisions.extend(paire) for paire in pairesCollisions]

        # Calcule le déplacement de analyseSphere et met à jour
        # lastPos pour l'utiliser à la prochaine itération
        deplacementVec: vp.vector = atoms[analyseSphere].pos - lastPos
        deplacementScalaire: float = vp.mag(deplacementVec)
        lastPos = atoms[analyseSphere].pos

        # Ajoute le temps écoulé et la distance parcourue pendant le temps dt
        # au total depuis la dernière collision
        dVec[-1] += deplacementVec
        dScalaire[-1] += deplacementScalaire
        tCollision[-1] += dt

        # Si la sphère est impliquée dans une collision, passer à la prochaine
        # valeur de temps et de distance entre collisions
        if analyseSphere in collisions:
            dVec.append(vp.vector(0, 0, 0))
            dScalaire.append(0)
            tCollision.append(0)


if __name__ == "__main__":
    main()
