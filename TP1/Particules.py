import vpython as vp


class Particule:
    def __init__(self, position: vp.vector, qteMvt: vp.vector, masse: float, rayon: float, couleur: vp.color):
        self.pos: vp.vector = position
        self.p: vp.vector = qteMvt
        self.masse: float = masse
        self.rayon: float = rayon
        self.sphere = vp.simple_sphere(self.pos, radius=rayon, color=couleur)


    def update(self, dt: float, L: float):
        # Fait un pas à la particule pour un temps dt
        vitesse: vp.vector = self.p/self.masse
        self.pos += vitesse * dt

        # Fait rebondir la particule lors d'une collision avec la boite
        if self.pos.x < -L/2 or self.pos.x > L/2:
            self.p.x = -self.p.x
        if self.pos.y < -L/2 or self.pos.y > L/2:
            self.p.y = -self.p.y

        # self.draw() Peut être pas nécessaire


    def draw(self):
        self.sphere.pos = self.pos


def collisionAtomes(atome1: Particule, atome2: Particule):
    pass
