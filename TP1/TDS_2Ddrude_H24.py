# Standard library
import random
from functools import partial

# Third-party libraries
import numpy as np
import vpython as vp
from scipy import constants as cte

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

# Déclaration des paramètres du cadrillage
num_ions = 10  # Nombre de sphères le long de l'axe y
espace_x = L / (num_ions - 1)  # Espacement équidistant entre les sphères en x
espace_y = L / (num_ions - 1)  # Espacement équidistant entre les sphères en y
R_ions = 2*Ratom
# Création des ions pour le cadrillage en 2D
ions = []
for i in range(num_ions):
    for j in range(num_ions):
        x = -L / 2 + i * espace_x
        y = -L / 2 + j * espace_y
        # Décalage de la position en y pour chaque deuxième rangée
        if j % 2 == 1:
            x += espace_x / 2
        # Vérification si les coordonnées de la sphère se trouvent à l'intérieur de la boîte
        if -L/2 <= x <= L/2 and -L/2 <= y <= L/2:
            ions.append(vp.sphere(pos=vp.vec(x, y, 0), radius=R_ions, color=vp.color.blue))


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
        sphere = simple_sphere(radius=0.02, color=vp.color.magenta)
    else: 
        sphere = simple_sphere(radius=Ratom, color=gray)

    Atoms.append(sphere)
    phi = 2 * np.pi * random.random()  # direction aléatoire pour la quantité de mouvement
    px = pavg * np.cos(phi)     # quantité de mvt initiale selon l'équipartition
    py = pavg * np.sin(phi)

    # liste de la quantité de mvt initiale de toutes les sphères
    p.append(vp.vec(px, py, pz))

# BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt
#for count in range(timeLoopLen):
while True:

    # limite la vitesse de calcul de la simulation pour que l'animation soit
    # visible à l'oeil humain!
    vp.rate(5)

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


    # Vérification des collisions avec les ions
        for ion in ions:
            # Calcul de la distance entre l'atome et l'ion
            distance = vp.mag(apos[i] - ion.pos)
            if distance < 2 * Ratom:  # Si l'atome touche l'ion
                # Génération d'une nouvelle direction aléatoire pour les composantes x et y de la quantité de mouvement
                phi = 2 * np.pi * random.random()  # Nouvelle direction aléatoire
                p[i].x = pavg * np.cos(phi)  # Nouvelle composante x
                p[i].y = pavg * np.sin(phi)  # Nouvelle composante y
