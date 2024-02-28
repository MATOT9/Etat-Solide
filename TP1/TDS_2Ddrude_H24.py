# Third-party libraries
from Particules import *
from scipy import constants as cte
import numpy as np
import sys
import vpython as vp
import matplotlib.pyplot as plt


def main(dt: float, timeLoopLen: int, analyseSphere: int, E: float = 0, rand: bool = True, render: bool = True) -> [list[Particule]]:
    # Déclaration de variables physiques "Typical values"
    mass: float = cte.m_e
    charge: float = -cte.e
    rElec: float = 0.01
    T: float = 300

    # Canevas de fond et arêtes de boîte 2d
    L: float = 1    # container is a cube L on a side
    if render:
        canvas (L, rElec)

    # win = 500 # peut aider à définir la taille d'un autre objet visuel comme un histogramme proportionnellement à la taille du canevas.
    # change this to have more or fewer atoms
    nElec: int = 200
    # Maille primitive du cristal
    maillePrimitive: list[vp.vector] = [vp.vector(3*L/10, 0, 0), np.sqrt(3)*L/10 * vp.vector(np.cos(np.pi/6), np.sin(np.pi/6), 0)]

    particule1: list[float, vp.vector] = [0.03, vp.vector(0, 0, 0)]
    particule2: list[float, vp.vector] = [0.03, vp.vector(L/10, 0, 0)]
    motif: list[list[float, vp.vector]] = [particule1, particule2]

    # Initialisation des atomes
    electrons: list[Particule] = setupParticules(L, mass, rElec, nElec, T, analyseSphere, charge=charge, rand=rand, render=render)
    coeurs: list[Particule] = setupCoeurs(maillePrimitive, motif, L, render=render)

    # BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt
    # Initialisation des listes 
    posMoy: list[vp.vector] = []
    pMoy: list[float] = []
    pAnalyseSphere: list[float] = []
    nbCollisions: int = 0

    for loop in range(timeLoopLen):
        # limite la vitesse de calcul de la simulation pour que l'animation soit
        # visible à l'oeil humain!
        if render:
            vp.rate(60)

        [electron.update(dt, L, E) for electron in electrons]

        # Données nécessaires pour les questions dans le notebook
        allPos = [electron.pos for electron in electrons] # vecteurs pos de l'ittération
        allP = [electron.p for electron in electrons] # vecteurs p de l'ittération
        posMoy.append(sum(allPos, vp.vector(0, 0, 0))/len(allPos))
        pMoy.append(vp.mag(sum(allP, vp.vector(0, 0, 0)))/len(allP))
        pAnalyseSphere.append(vp.mag(allP[analyseSphere]))

        # LET'S FIND THESE COLLISIONS!!!
        hitlist: list[list[int]] = Particule.checkCollisionsElectrons(electrons, coeurs)
        # On compte le nombre total de collisions
        nbCollisions += len(hitlist)

        # Calcule le résultat des collisions et bouge les atomes
        for i, j in hitlist:
            Particule.collisionsElectrons(electrons[i], coeurs[j], T)
        
    
    return posMoy, pMoy, pAnalyseSphere, timeLoopLen*dt*nElec/nbCollisions


if __name__ == "__main__":
    # Capture les arguments de la ligne de commande pour contrôler le champ électrique
    # la direction initiale des particules et si la simulation est affichée
    E: float = float(sys.argv[1])
    rand: bool = bool(int(sys.argv[2]))
    render: bool = bool(int(sys.argv[3]))
    dt: float = 2e-8         # pas d'incrémentation temporel
    timeLoopLen: int = 100  # temps de simulation
    analyseSphere: int = 56 # à changer si analyse autre atome He (int [0, 199])
    tVector = np.linspace(0, dt*(timeLoopLen-1), timeLoopLen) #vecteur de temps
    posMoy, pMoy, pAnalysSphere, tau = main(dt, timeLoopLen, analyseSphere, E=E, rand=rand, render=render)

    print(posMoy)
    print(pMoy)
    print(tau)
