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
from Particules import *
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
            atoms.append(particule(vp.color.magenta))
            atoms[-1].sphere.radius *= 2
        else: 
            atoms.append(particule(gray))

    # BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt
    for _ in range(timeLoopLen):
        # limite la vitesse de calcul de la simulation pour que l'animation soit
        # visible à l'oeil humain!
        vp.rate(300)

        [atom.update(dt, L) for atom in atoms]

        # LET'S FIND THESE COLLISIONS!!!
        hitlist: list[list[int]] = checkCollisions(atoms)

        # Calcule le résultat des collisions et bouge les atomes
        for i, j in hitlist:
            collisionAtomes(atoms[i], atoms[j])

        # Met à jour la liste des variables à suivre
        # Unpack les paires de collisions
        collisions: list[int] = []
        [collisions.extend(paire) for paire in hitlist]

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
    
    return atoms, dVec, dScalaire, tCollision, analyseSphere, dt, timeLoopLen


if __name__ == "__main__":
    atoms, dVec, dScalaire, tCollision, analyseSphere, dt, timeLoopLen = main()
