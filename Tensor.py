import Vector;
import math
import stat

# TODO: List to Matrix as Array

class Tensor:
    # TODO: Private methods
    # private oldTheta: boolean
    # private _theta: number

    def __init__(self, r, matrix):
        # Represent the matrix as a 2 element list
        # [ 0, 1
        #   1, -0 ]
        # constructor(private r: number, private matrix: number[]) { ... } 
        self.oldTheta = False
        self._theta = self.calculateTheta()


    @staticmethod
    def fromAngle(angle):
        return Tensor(1, [math.cos(angle * 4), math.sin(angle * 4)])


    @staticmethod
    def fromVector(vector):
        t1 = vector.x ** 2 - vector.y ** 2
        t2 = 2 * vector.x * vector.y
        t3 = t1 ** 2 - t2 ** 2
        t4 = 2 * t1 * t2
        return Tensor(1, [t3, t4])


    @staticmethod
    def get_zero(): return Tensor(0, [0,0])

    def get_theta(self): 
        if (self.oldTheta):
            self._theta = self.calculateTheta()
            self.oldTheta = False
        return self._theta

    def add(self, tensor, smooth):
        # self.matrix = self.matrix.map((v, i) => v * self.r + tensor.matrix[i] * tensor.r)
        self.matrix = [self.matrix[i] * self.r + tensor.matrix[i] * tensor.r for i in range(self.matrix)]

        if (smooth):
            # self.r = math.hypot(...self.matrix)
            self.r = math.hypot(*self.matrix)
            # self.matrix = self.matrix.map(v => v / self.r)
            self.matrix = [v/self.r for v in self.matrix]
        else: self.r = 2

        self.oldTheta = True
        return self


    def scale(self, s):
        self.r *= s
        self.oldTheta = True
        return self


    # Radians
    def rotate(self, theta):
        if theta == 0: return self

        newTheta = self.theta + theta
        if (newTheta < math.PI): newTheta += math.PI
        if (newTheta >= math.PI): newTheta -= math.PI

        self.matrix[0] = math.cos(2 * newTheta) * self.r
        self.matrix[1] = math.sin(2 * newTheta) * self.r
        self._theta = newTheta

        return self

    
    def getMajor(self): 
        # Degenerate case
        if (self.r == 0): return Vector.zeroVector()
        return Vector(math.cos(self.theta), math.sin(self.theta))

    def getMinor(self):
        # Degenerate case
        if (self.r == 0): return Vector.zeroVector()

        angle = self.theta + math.PI / 2
        return Vector(math.cos(angle), math.sin(angle))


    # Private
    def __calculateTheta(self):
        if self.r == 0: return 0
        return math.atan2(self.matrix[1] / self.r, self.matrix[0] / self.r) / 2