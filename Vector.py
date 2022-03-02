# TODO: Import logger
import math
import stat

class Vector:
    def __init__(self, x, y):
        # constructor(public x: number, public y: number) {}
        self.x = x
        self.y = y
    
    def add(self, v):
        self.x += v.x
        self.y += v.y
        return self
    
    # Angle in radians to positive x-axis between -pi and pi
    def angle(self):
        return math.atan2(self.y, self.x)

    def clone(self):
        return Vector(self.x, self.y)

    def copy(self, v):
        self.x = v.x
        self.y = v.y
        return self
    
    def cross(self, v):
        return self.x * v.y - self.y * v.x

    def distanceTo(self, v):
        return math.sqrt(self.distanceToSquared(v))

    def distanceToSquared(self, v):
        dx = self.x - v.x
        dy = self.y - v.y
        return dx * dx + dy * dy

    def divide(self, v):
        if (v.x == 0 or v.y == 0):
            print("Division by zero")
            return self

        self.x /= v.x
        self.y /= v.y
        return self

    def divideScalar(self, s):
        if (s == 0):
            print("Division by zero")
            return self
        
        return self.multiplyScalar(1 / s)

    def dot(self, v):
        return self.x * v.x + self.y * v.y

    def equals(self, v):
        return ((v.x == self.x) and (v.y == self.y))

    def length(self):
        return math.sqrt(self.lengthSq())

    def lengthSq(self):
        return self.x * self.x + self.y * self.y

    def multiply(self, v):
        self.x *= v.x
        self.y *= v.y
        return self

    def multiplyScalar(self, s):
        self.x *= s
        self.y *= s
        return self

    def negate(self):
        return self.multiplyScalar(-1)

    def normalize(self):
        l = self.length()
        if (l == 0):
            print("Zero Vector")
            return self
        return self.divideScalar(l)

    # Angle in radians
    def rotateAround(self, center, angle):
        cos = math.cos(angle)
        sin = math.sin(angle)

        x = self.x - center.x
        y = self.y - center.y

        self.x = x * cos - y * sin + center.x
        self.y = x * sin + y * cos + center.y
        return self

    def set(self, v):
        self.x = v.x
        self.y = v.y
        return self

    def setX(self, x):
        self.x = x
        return self

    def setY(self, y):
        self.y = y
        return self

    def setLength (self, length):
        return self.normalize().multiplyScalar(length)

    def sub(self, v):
        self.x -= v.x
        self.y -= v.y
        return self

    @staticmethod
    def zeroVector(): return Vector(0, 0)

    @staticmethod
    def fromScalar(s): return Vector(s, s)

    @staticmethod
    def angleBetween(v1, v2):
        angleBetween = v1.angle() - v2.angle()
        if (angleBetween > math.pi): angleBetween -= 2 * math.pi
        elif (angleBetween <= -math.pi): angleBetween += 2 * math.pi
        return angleBetween

    @staticmethod
    # Tests whether a point lies to the left of a line
    def isLeft(linePoint, lineDirection, point):
        perpendicularVector = Vector(lineDirection.y, -lineDirection.x)
        return bool(point.clone().sub(linePoint).dot(perpendicularVector) < 0)

# Vector.zeroVector = staticmethod(Vector.zeroVector)
# Vector.fromScalar = staticmethod(Vector.fromScalar)
# Vector.angleBetween = staticmethod(Vector.angleBetween)
# Vector.isLeft = staticmethod(Vector.isLeft)
