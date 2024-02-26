import vpython as vp


class Particule:
    def __init__(self, position: vp.vector, qteMvt: vp.vector, masse: float, rayon: float, couleur: vp.color):
        self.pos: vp.vector = position
        self.p: vp.vector = qteMvt
        self.masse: float = masse
        self.rayon: float = rayon
        self.sphere = vp.simple_sphere(pos=position, radius=rayon, color=couleur)


    def update(self, dt: float, L: float):
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
        self.draw()


    def draw(self):
        self.sphere.pos = self.pos


def collisionAtomes(atome1: Particule, atome2: Particule):
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
