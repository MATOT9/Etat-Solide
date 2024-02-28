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
    nElec: int = 500
    # Maille primitive du cristal
    maillePrimitive: list[vp.vector] = [vp.vector(3*L/10, 0, 0), np.sqrt(3)*L/10 * vp.vector(np.cos(np.pi/6), np.sin(np.pi/6), 0)]

    particule1: list[float, vp.vector] = [0.03, vp.vector(0, 0, 0)]
    particule2: list[float, vp.vector] = [0.03, vp.vector(L/10, 0, 0)]
    motif: list[list[float, vp.vector]] = [particule1, particule2]

    # Initialisation des atomes
    electrons: list[Particule] = setupParticules(L, mass, rElec, nElec, T, analyseSphere, charge=charge, rand=rand, render=render)
    coeurs: list[Particule] = setupCoeurs(maillePrimitive, motif, L, render=render)

    # BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt
    pMoy, pAnalysSphere = np.zeros(timeLoopLen), np.zeros(timeLoopLen) #initialise les array question 2
    posx, posy = np.zeros(timeLoopLen), np.zeros(timeLoopLen) #initalise liste de positions question 4
    for loop in range(timeLoopLen):
        # limite la vitesse de calcul de la simulation pour que l'animation soit
        # visible à l'oeil humain!
        if render:
            vp.rate(60)

        [electron.update(dt, L, E) for electron in electrons]
        allP = [electron.p for electron in electrons] # vecteurs p de l'ittération
        allPos = [electron.pos for electron in electrons] # vecteurs pos de l'ittération
        sumP = vp.vector(0, 0 ,0)
        #pMoy[loop] = sum(allP)/len(allP)
        for P in allP:
            sumP += P
        pMoy[loop] = (sumP.x**2+sumP.y**2)**0.5/len(allP)
        pAS = allP[analyseSphere]
        pAnalysSphere[loop] = (pAS.x**2+pAS.y**2)**0.5

        # LET'S FIND THESE COLLISIONS!!!
        hitlist: list[list[int]] = Particule.checkCollisionsElectrons(electrons, coeurs)

        # Calcule le résultat des collisions et bouge les atomes
        for i, j in hitlist:
            Particule.collisionsElectrons(electrons[i], coeurs[j], T)
        
    
    return electrons, pMoy, pAnalysSphere


if __name__ == "__main__":
    # Capture les arguments de la ligne de commande pour contrôler le champ électrique
    # la direction initiale des particules et si la simulation est affichée
    E: float = float(sys.argv[1])
    rand: bool = bool(int(sys.argv[2]))
    render: bool = bool(int(sys.argv[3]))
    dt: float = 2e-8         # pas d'incrémentation temporel
    timeLoopLen: int = 1000  # temps de simulation
    analyseSphere: int = 56 # à changer si analyse autre atome He (int [0, 199])
    tVector = np.linspace(0, dt*(timeLoopLen-1), timeLoopLen) #vecteur de temps
    electrons, pMoy, pAnalysSphere = main(dt, timeLoopLen, analyseSphere, E=E, rand=rand, render=render)
