import math
import Vector
import Tensor
from basis_field import Grid, Radial, BasisField

# import * as noise from 'noisejs'
# import * as SimplexNoise from 'simplex-noise'
from polygon_util import PolygonUtil

NoiseParams = { 
    'globalNoise': 0, 
    'noiseSizePark': 0, 
    'noiseAnglePark': 0, # Degrees
    'noiseSizeGlobal': 0, 
    'noiseAngleGlobal': 0
}

# export interface NoiseParams {
#     globalNoise: boolean
#     noiseSizePark: number
#     noiseAnglePark: number  // Degrees
#     noiseSizeGlobal: number
#     noiseAngleGlobal: number
# }

# Combines basis fields
# TODO Noise added when sampling a point in a park
class TensorField:
    basisFields = []
    # private basisFields: BasisField[] = []
    # private noise: SimplexNoise

    # public parks: Vector[][] = []
    # public sea: Vector[] = []
    # public river: Vector[] = []
    # public ignoreRiver = False

    smooth = False

    def __init__(self, noiseParams):
        # constructor(public noiseParams: NoiseParams)
        # TODO self.noise = SimplexNoise()
        pass

    # Used when integrating coastline and river
    # TODO def enableGlobalNoise(angle, size):
        # self.noiseParams.globalNoise = True
        # self.noiseParams.noiseAngleGlobal = angle
        # self.noiseParams.noiseSizeGlobal = size

    # def disableGlobalNoise(): self.noiseParams.globalNoise = False

    def __addField(self, field): self.basisFields.push(field)

    def addGrid(self, centre, size, decay, theta):
        grid = Grid(centre, size, decay, theta)
        self.__addField(grid)

    def addRadial(self, centre, size, decay):
        radial = Radial(centre, size, decay)
        self.__addField(radial)

    # TODO
    def __removeField(self, field):
        index = self.basisFields.indexOf(field)
        if (index > -1): self.basisFields.splice(index, 1)

    def reset(self):
        self.basisFields = []
        # TODO
        # self.parks = []
        # self.sea = []
        # self.river = []

    def getCentrePoints(self): return [field.centre for field in self.basisFields]

    def getBasisFields(self): return self.basisFields

    def samplePoint(self, point):
        # Degenerate point
        if not self.onLand(point): return Tensor.zero

        # Default field is a grid
        if len(self.basisFields) == 0: return Tensor(1, [0, 0])

        tensorAcc = Tensor.zero
        for field in self.basisFields:
            tensorAcc.add(field.getWeightedTensor(point, self.smooth), self.smooth)

        # TODO
        # Add rotational noise for parks - range -pi/2 to pi/2
        # if (self.parks.some(p => PolygonUtil.insidePolygon(point, p))): tensorAcc.rotate(self.getRotationalNoise(point, self.noiseParams.noiseSizePark, self.noiseParams.noiseAnglePark))
            # TODO optimise insidePolygon e.g. distance
        # if (self.noiseParams.globalNoise): tensorAcc.rotate(self.getRotationalNoise(point, self.noiseParams.noiseSizeGlobal, self.noiseParams.noiseAngleGlobal))

        return tensorAcc

    # Noise Angle is in degrees
    def getRotationalNoise(self, point, noiseSize, noiseAngle):
        return self.noise.noise2D(point.x / noiseSize, point.y / noiseSize) * noiseAngle * math.PI / 180

    # TODO
    def onLand(self, point):
        # inSea = PolygonUtil.insidePolygon(point, self.sea)
        # if self.ignoreRiver: return not inSea
        # return not inSea and not PolygonUtil.insidePolygon(point, self.river)

        # TODO temporarily
        return True

    # TODO
    def inParks(self, point):
        for p in self.parks:
            if (PolygonUtil.insidePolygon(point, p)): return True
        return False
