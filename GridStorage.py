import Vector
import math

# Cartesian grid accelerated data structure
# Grid of cells, each containing a list of vectors
class GridStorage:
    # TODO:
    # private gridDimensions: Vector
    # private grid: Vector[][][]
    # private dsepSq: number


    # worldDimensions assumes origin of 0,0
    # @param {number} dsep Separation distance between samples
    def __init__(self, worldDimensions, origin, dsep):
        # constructor (private worldDimensions: Vector, private origin: Vector, private dsep: number)
        self.dsepSq = self.dsep * self.dsep
        self.gridDimensions = worldDimensions.clone().divideScalar(self.dsep)
        self.grid = []
        for x in range(self.gridDimensions.x):
            self.grid.push([])
            for y in range(self.gridDimensions.y):
                self.grid[x].push([])

    # Add all samples from another grid to this one
    def addAll(self, gridStorage):
        for row in gridStorage.grid:
            for cell in row:
                for sample in cell:
                    self.addSample(sample)

    def addPolyline(self, line):
        for v in line:
            self.addSample(v)
        
    

    # Does not enforce separation
    # Does not clone
    def addSample(self, v, coords):
        if not coords:
            coords = self.getSampleCoords(v)
        self.grid[coords.x][coords.y].push(v)


    #  Tests whether v is at least d away from samples
    #  Performance very important - this is called at every integration step
    #  @param dSq=self.dsepSq squared test distance
    #  Could be dtest if we are integrating a streamline
    def isValidSample(self, v, dSq=None):
        if not dSq:
            dSq = self.dsepSq
        
        # Code duplication with self.getNearbyPoints but much slower when calling
        # self.getNearbyPoints due to array creation in that method
        
        coords = self.getSampleCoords(v)

        # Check samples in 9 cells in 3x3 grid
        for x in range(-1,2):
        # for (let x = -1 x <= 1 x++) {
            for y in range(-1,2):
            # for (let y = -1 y <= 1 y++) {
                cell = coords.clone().add(Vector(x, y))
                if not self.vectorOutOfBounds(cell, self.gridDimensions):
                    if not self.vectorFarFromVectors(v, self.grid[cell.x][cell.y], dSq): return False
        return True
    

    # Test whether v is at least d away from vectors
    # Performance very important - this is called at every integration step
    # @param {number}   dSq     squared test distance
    def vectorFarFromVectors(v, vectors, dSq):
        for sample in vectors:
            if sample != v:
                distanceSq = sample.distanceToSquared(v)
                if (distanceSq < dSq):
                    return False
        return True

    # Returns points in cells surrounding v
    # Results include v, if it exists in the grid
    # @param {number} returns samples (kind of) closer than distance - returns all samples in 
    # cells so approximation (square to approximate circle)
    def getNearbyPoints(self, v, distance):
        radius = math.ceil((distance/self.dsep) - 0.5)
        coords = self.getSampleCoords(v)
        out = []
        # TODO: const out: Vector[] = []
        for x in range(-1 * radius, 1 * radius):
            for y in range(-1 * radius, 1 * radius):
                cell = coords.clone().add(Vector(x, y))
                if not self.vectorOutOfBounds(cell, self.gridDimensions):
                    for v2 in self.grid[cell.x][cell.y]:
                        out.push(v2)
        return out

    def worldToGrid(self, v): return v.clone().sub(self.origin)  

    def gridToWorld(self, v): return v.clone().add(self.origin)

    def vectorOutOfBounds(gridV, bounds):
        return (gridV.x < 0 or gridV.y < 0 or gridV.x >= bounds.x or gridV.y >= bounds.y)

    # @return {Vector}   Cell coords corresponding to vector
    # Performance important - called at every integration step
    def getSampleCoords(self, worldV):
        v = self.worldToGrid(worldV)
        if (self.vectorOutOfBounds(v, self.worldDimensions)): return Vector.zeroVector()
        #  log.error("Tried to access out-of-bounds sample in grid")

        return Vector( math.floor(v.x / self.dsep), math.floor(v.y / self.dsep) )
