"""
Created on Fri Feb  5 15:40:27 2021

#GlowScript 3.0 VPython

# Hard-sphere gas.

# Bruce Sherwood
# Claudine Allen
"""

# Standard library
import random
from functools import partial

# Third-party libraries
import numpy as np
import vpython as vp
from scipy import constants as cte


# win = 500 # peut aider à définir la taille d'un autre objet visuel comme un histogramme proportionnellement à la taille du canevas.
# Déclaration de variables influençant le temps d'exécution de la simulation
Natoms = 200    # change this to have more or fewer atoms
dt = 1e-5       # pas d'incrémentation temporel

# variable à ajouter Partie 1 num 3
timeLoopLen = 1000  # temps de simulation
analyseSphere = 106  # à changer si analyse autre atome He (int [0, 199])
histPosSphereX = histPosSphereY = np.zeros(timeLoopLen) # va permettre d'enregistrer toutes les poistions pour obtenir la vitesse en composantes
distanceCollision = []  # vecteur de distance parcourrue
timeCollision = []  # vecteur de temps écoulé

# Déclaration de variables physiques "Typical values"
mass = cte.value("alpha particle mass")
Ratom = 0.01
T = 300

# CANEVAS DE FOND
L = 1                       # container is a cube L on a side
gray = vp.color.gray(0.7)   # color of edges of container and spheres below
animation = vp.canvas(width=750, height=500)
animation.range = L
animation.title = 'Cinétique des gaz parfaits'

# ARÊTES DE BOÎTE 2D
d = L/2 + Ratom
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
Atoms = []  # Objet qui contiendra les sphères pour l'animation
p = []      # quantité de mouvement des sphères
apos = []   # position des sphères

# Principe de l'équipartition de l'énergie en thermodynamique statistique classique
pavg = (2 * mass * 1.5 * cte.k * T)**0.5

for i in range(Natoms):
    # position aléatoire qui tient compte que l'origine est au centre de la boîte
    xyz = L * np.random.rand(3) - L/2
    xyz[-1] = pz = 0

    # liste de la position initiale de toutes les sphères
    apos.append(vp.vec(*xyz))

    simple_sphere = partial(vp.simple_sphere, pos=vp.vec(*xyz))
    if i == 0:
        # garde une sphère plus grosse et colorée parmis toutes les grises
        sphere = simple_sphere(radius=0.03, color=vp.color.magenta)
    else: 
        sphere = simple_sphere(radius=Ratom, color=gray)

    Atoms.append(sphere)
    phi = 2 * np.pi * random.random()  # direction aléatoire pour la quantité de mouvement
    px = pavg * np.cos(phi)     # quantité de mvt initiale selon l'équipartition
    py = pavg * np.sin(phi)

    # liste de la quantité de mvt initiale de toutes les sphères
    p.append(vp.vec(px, py, pz))


# FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
def checkCollisions():
    hitlist = []

    # distance critique où les 2 sphères entre en contact à la limite de leur rayon
    r2 = (2 * Ratom)**2

    for i in range(Natoms):
        ai = apos[i]
        for j in range(i) :
            aj = apos[j]

            # la boucle dans une boucle itère pour calculer cette distance
            # vectorielle dr entre chaque paire de sphère
            dr = ai - aj
            if vp.mag2(dr) < r2:
                # liste numérotant toutes les paires de sphères en collision
                hitlist.append([i,j])
    return hitlist

def tracingParticule(sphere, List, count):
    # fonction ajoutee pour track la particule
    l1 = l2 = np.empty(len(List))
    for i, _ in enumerate(l1):
        l1[i] = List[i][0]
        l2[i] = List[i][1]

    timeCollision[-1] += dt
    varX = histPosSphereX[count] - histPosSphereX[count - 1]
    varY = histPosSphereY[count] - histPosSphereY[count - 1]
    distanceCollision[-1] += np.linalg.norm(varX + varY)
    if sphere in l1 or sphere in l2:
        distanceCollision.append(0)
        timeCollision.append(0)


# BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt
for count in range(timeLoopLen):
    # limite la vitesse de calcul de la simulation pour que l'animation soit
    # visible à l'oeil humain!
    vp.rate(300)

    # DÉPLACE TOUTES LES SPHÈRES D'UN PAS SPATIAL deltax
    vitesse = []    # vitesse instantanée de chaque sphère
    deltax = []     # pas de position des sphère (incrément de temps dt)
    for i in range(Natoms):
        vitesse.append(p[i] / mass)
        deltax.append(vitesse[i] * dt)

        # nouvelle position de l'atome après l'incrément de temps dt
        Atoms[i].pos = apos[i] = apos[i] + deltax[i]

        # CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS AVEC LES PAROIS DE LA BOÎTE
    # for i in range(Natoms):
        loc = apos[i]
        if abs(loc.x) > L/2 and loc.x < 0:
            # renverse composante x à la paroi de gauche
            p[i].x = abs(p[i].x)
        elif abs(loc.x) > L/2:
            # renverse composante x à la paroi de droite
            p[i].x = -abs(p[i].x)

        if abs(loc.y) > L/2 and loc.y < 0:
            # renverse composante y à la paroi du bas
            p[i].y = abs(p[i].y)
        elif abs(loc.y) > L/2:
            # renverse composante y à la paroi du haut
            p[i].y = -abs(p[i].y)

    # LET'S FIND THESE COLLISIONS!!!
    hitlist = checkCollisions()
    histPosSphereX[count] = apos[analyseSphere].x
    histPosSphereY[count] = apos[analyseSphere].y 
    if count != 0:
        tracingParticule(analyseSphere, hitlist, count)
    else:
        distanceCollision, timeCollision  = [0], [0]

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

        # exclusion de cas où il n'y a pas de changements à faire
        if vrel.mag2 == 0 or rrel.mag > Ratom:
            continue

        # calcule la distance et temps d'interpénétration des sphères dures qui ne doit pas se produire dans ce modèle
        dx = vp.dot(rrel, vrel.hat)
        dy = vp.cross(rrel, vrel.hat).mag
        alpha = vp.asin(dy / (2 * Ratom))  # alpha is the angle of the triangle composed of rrel, path of atom j, and a line from the center of atom i to the center of atom j where atome j hits atom i
        d = (2 * Ratom) * vp.cos(alpha) - dx # distance traveled into the atom from first contact
        deltat = d / vrel.mag         # time spent moving from first contact to position inside atom

        # CHANGE L'INTERPÉNÉTRATION DES SPHÈRES PAR LA CINÉTIQUE DE COLLISION
        posi = posi-vi*deltat
        posj = posj-vj*deltat

        # transform momenta to center-of-momentum (com) frame
        pcomi = p[i]-mass*Vcom
        pcomj = p[j]-mass*Vcom

        # bounce in center-of-momentum (com) frame
        pcomi = pcomi - 2 * vp.dot(pcomi, rrel) * rrel.hat
        pcomj = pcomj - 2 * vp.dot(pcomj, rrel) * rrel.hat

        # transform momenta back to lab frame
        p[i] = pcomi + mass * Vcom
        p[j] = pcomj + mass * Vcom

        # move forward deltat in time, ramenant au même temps où sont rendues
        # les autres sphères dans l'itération
        apos[i] = posi + (p[i] / mass) * deltat
        apos[j] = posj + (p[j] / mass) * deltat
