"""
Created on Fri Feb  5 15:40:27 2021

#GlowScript 3.0 VPython

# Hard-sphere gas.

# Bruce Sherwood
# Claudine Allen
"""

# Third-party libraries
from Particules import *
from scipy import constants as cte
import numpy as np
import sys
import vpython as vp


def main(dt: float, timeLoopLen: int, analyseSphere: int) -> [list[Particule], list[vp.vector], list[float], list[float]]:
    # win = 500 # peut aider à définir la taille d'un autre objet visuel comme un histogramme proportionnellement à la taille du canevas.
    nAtoms: int = 200    # change this to have more or fewer atoms

    # Variables reliées à l'atome à étudier
    # Listes des variables suivies entre les collisions
    dVec: list[vp.vector] = [vp.vector(0, 0, 0)]
    dScalaire: list[float] = [0]
    tCollision: list[float] = [0]

    # Moyenne des normes des vitesses pour une itération pour le TCL
    S_n: list[float] = []

    # Déclaration de variables physiques "Typical values"
    mass: float = cte.value("alpha particle mass")
    rAtom: float = 0.01
    T: float = 300

    # Canevas de fond et arêtes de boîte 2d
    L: float = 1    # container is a cube L on a side
    canvas (L, rAtom)

    # Initialisation des atomes
    atoms: list[Particule] = setupParticules(L, mass, rAtom, nAtoms, T, analyseSphere)
    lastPos: vp.vector = atoms[analyseSphere].pos

    # BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt
    for _ in range(timeLoopLen):
        # limite la vitesse de calcul de la simulation pour que l'animation soit
        # visible à l'oeil humain!
        vp.rate(300)

        [atom.update(dt, L) for atom in atoms]

        # LET'S FIND THESE COLLISIONS!!!
        hitlist: list[list[int]] = Particule.checkCollisionsAtomes(atoms)

        # Calcule le résultat des collisions et bouge les atomes
        for i, j in hitlist:
            Particule.collisionAtomes(atoms[i], atoms[j])

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

        # On fait la somme des magnitudes des vitesses des 200 atomes
        S_i: float = sum([vp.mag(atom.p)/mass for atom in atoms])
        S_n.append(S_i)
    
    return atoms, dVec, dScalaire, tCollision, S_n


if __name__ == "__main__":
    timeLoopLen: int = int(sys.argv[1]) # Nombre d'itérations
    dt: float = 1e-5         # pas d'incrémentation temporel
    analyseSphere: int = 106 # à changer si analyse autre atome He (int [0, 199])
    atoms, dVec, dScalaire, tCollision, S_n = main(dt, timeLoopLen, analyseSphere)
