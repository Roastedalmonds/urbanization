from numpy import Infinity, number
import Vector
import GridStorage
import math
from Integrator import FieldIntegrator
from Simplify import simplify
from promise import Promise

class StreamlineIntegration:
    def __init__(self, seed, originalDir, streamline, previousDirection, previousPoint, valid):
        self.seed = seed
        self.originalDir = originalDir
        self.streamline = streamline
        self.previousDirection = previousDirection
        self.previousPoint = previousPoint
        self.valid = valid


class StreamlineParams:
    def __init__(self, dsep, dtest, dstep, dcirclejoin, dlookahead, joinangle, pathIterations, seedTries, simplifyTolerance, collideEarly):
        # dsep,  # Streamline seed separating distance
        self.dsep = dsep
        # dtest,  # Streamline integration separating distance
        self.dtest =  dtest
        # dstep,  # Step size
        self.dstep =  dstep
        # dcirclejoin,  # How far to look to join circles - (e.g. 2 x dstep)
        self.dcirclejoin =  dcirclejoin
        # dlookahead,  # How far to look ahead to join up dangling
        self.dlookahead =  dlookahead
        # joinangle,  # Angle to join roads in radians
        self.joinangle =  joinangle
        # pathIterations,  # Path integration iteration limit
        self.pathIterations =  pathIterations
        # seedTries,  # Max failed seeds
        self.seedTries =  seedTries
        self.simplifyTolerance =  simplifyTolerance
        # collideEarly  # Chance of early collision 0-1
        self.collideEarly = collideEarly


# Creates polylines that make up the roads by integrating the tensor field
class StreamlineGenerator:
    SEED_AT_ENDPOINTS = False
    NEAR_EDGE = 3  # Sample near edge

    majorGrid: GridStorage
    minorGrid: GridStorage
    paramsSq: StreamlineParams

    # How many samples to skip when checking streamline collision with itself
    nStreamlineStep: int
    # How many samples to ignore backwards when checking streamline collision with itself
    nStreamlineLookBack: int
    dcollideselfSq: int

    candidateSeedsMajor: Vector = []
    candidateSeedsMinor: Vector = []
    streamlinesDone = True
    # resolve: () => void
    lastStreamlineMajor = True

    allStreamlines: Vector = []
    streamlinesMajor: Vector = []
    streamlinesMinor: Vector = []
    allStreamlinesSimple: Vector = []  # Reduced vertex count


    # Uses world-space coordinates
    def __init__(self, integrator, origin, worldDimensions, params):
        # constructor(protected integrator: FieldIntegrator, protected origin: Vector, protected worldDimensions: Vector, protected params: StreamlineParams)
        if params.dstep > params.dsep: print("ERROR: STREAMLINE SAMPLE DISTANCE BIGGER THAN DSEP")

        # Enforce test < sep
        params.dtest = math.min(params.dtest, params.dsep)

        # Needs to be less than circlejoin
        self.dcollideselfSq = (params.dcirclejoin / 2) ** 2
        self.nStreamlineStep = math.floor(params.dcirclejoin / params.dstep)
        self.nStreamlineLookBack = 2 * self.nStreamlineStep

        self.majorGrid = GridStorage(self.worldDimensions, self.origin, params.dsep)
        self.minorGrid = GridStorage(self.worldDimensions, self.origin, params.dsep)

        self.setParamsSq()


    def clearStreamlines(self):
        self.allStreamlinesSimple = []
        self.streamlinesMajor = []
        self.streamlinesMinor = []
        self.allStreamlines = []


    # Edits streamlines
    def joinDanglingStreamlines(self):
        # TODO do in update method
        for major in [True, False]:
            for streamline in self.streamlines(major):
                # Ignore circles
                if (streamline[0].equals(streamline[streamline.length - 1])): continue

                newStart = self.getBestNextPoint(streamline[0], streamline[4], streamline)
                if newStart != None:
                    for p in self.pointsBetween(streamline[0], newStart, self.params.dstep):
                        streamline.unshift(p)
                        self.grid(major).addSample(p)

                newEnd = self.getBestNextPoint(streamline[streamline.length - 1], streamline[streamline.length - 4], streamline)
                if newEnd != None:
                    for p in self.pointsBetween(streamline[streamline.length - 1], newEnd, self.params.dstep):
                        streamline.push(p)
                        self.grid(major).addSample(p)

        # Reset simplified streamlines
        self.allStreamlinesSimple = []
        for s in self.allStreamlines: self.allStreamlinesSimple.push(self.simplifyStreamline(s))


    # Returns array of points from v1 to v2 such that they are separated by at most dsep not including v1
    def pointsBetween(self, v1, v2, dstep):
        d = v1.distanceTo(v2)
        nPoints = math.floor(d / dstep)
        if nPoints == 0: return []

        stepVector = v2.clone().sub(v1)

        out = []
        i = 1
        next = v1.clone().add(stepVector.clone().multiplyScalar(i / nPoints))
        for i in range(nPoints+1):
            # Test for degenerate point
            if self.integrator.integrate(next, True).lengthSq() > 0.001: out.push(next)
            else: return out
            next = v1.clone().add(stepVector.clone().multiplyScalar(i / nPoints))
        return out

    # Gets next best point to join streamline returns None if there are no good candidates
    def getBestNextPoint(self, point, previousPoint, streamline):
        nearbyPoints = self.majorGrid.getNearbyPoints(point, self.params.dlookahead)
        nearbyPoints.push(*self.minorGrid.getNearbyPoints(point, self.params.dlookahead))
        direction = point.clone().sub(previousPoint)

        closestSample = None
        closestDistance = Infinity

        for sample in nearbyPoints:
            if (not sample.equals(point) and not sample.equals(previousPoint)): # and not streamline.includes(sample)):
                differenceVector = sample.clone().sub(point)
                if differenceVector.dot(direction) < 0: continue # Backwards
                
                # Acute angle between vectors (agnostic of CW, ACW)
                distanceToSample = point.distanceToSquared(sample)
                if (distanceToSample < 2 * self.paramsSq.dstep):
                    closestSample = sample
                    break

                angleBetween = math.abs(Vector.angleBetween(direction, differenceVector))

                # Filter by angle
                if (angleBetween < self.params.joinangle and distanceToSample < closestDistance):
                    closestDistance = distanceToSample
                    closestSample = sample

        # TODO is reimplement simplify-js to preserve intersection points
        #  - self is the primary reason polygons aren't found
        # If trying to find intersections in the simplified graph
        # prevent ends getting pulled away from simplified lines
        if closestSample != None: closestSample = closestSample.clone().add(direction.setLength(self.params.simplifyTolerance * 4))
        
        return closestSample

    # Assumes s has already generated
    def addExistingStreamlines(self, s):
        self.majorGrid.addAll(s.majorGrid)
        self.minorGrid.addAll(s.minorGrid)

    def setGrid(self, s):
        self.majorGrid = s.majorGrid
        self.minorGrid = s.minorGrid

    # returns True if state updates
    def update(self):
        if not self.streamlinesDone:
            self.lastStreamlineMajor = not self.lastStreamlineMajor
            if not self.createStreamline(self.lastStreamlineMajor):
                self.streamlinesDone = True
                self.resolve()
            return True
        return False

    # TODO: Promise
    # All at once - will freeze if dsep small
    async def createAllStreamlines(self, animate=False):
        # self.resolve = resolve
        self.streamlinesDone = False
        if not animate:
            major = True
            while self.createStreamline(major): major = not major
        self.joinDanglingStreamlines()
    # async createAllStreamlines(animate=False): Promise<void> {
    #     return Promise<void>(resolve => {
    #         self.resolve = resolve
    #         self.streamlinesDone = False
            # if
    #     }).then(() => self.joinDanglingStreamlines())
    # }

    def simplifyStreamline(self, streamline):
        simplified = []
        for point in simplify(streamline, self.params.simplifyTolerance): simplified.push(Vector(point.x, point.y))
        return simplified

    # Finds seed and creates a streamline from that point
    # Pushes candidate seeds to queue
    # @return {Vector[]} returns False if seed isn't found within params.seedTries
    def createStreamline(self, major):
        seed = self.getSeed(major)
        if seed == None: return False
        streamline = self.integrateStreamline(seed, major)
        if (self.validStreamline(streamline)):
            self.grid(major).addPolyline(streamline)
            self.streamlines(major).push(streamline)
            self.allStreamlines.push(streamline)

            self.allStreamlinesSimple.push(self.simplifyStreamline(streamline))

            # Add candidate seeds
            if not streamline[0].equals(streamline[streamline.length - 1]):
                self.candidateSeeds(not major).push(streamline[0])
                self.candidateSeeds(not major).push(streamline[streamline.length - 1])

        return True

    def validStreamline(self, s): return s.length > 5

    def setParamsSq(self):
        self.paramsSq = self.params.copy()
        for p in self.paramsSq:
            if (type(self.paramsSq[p] == "float")): self.paramsSq[p] *= self.paramsSq[p]


    # TODO better seeding scheme
    def samplePoint(self): return Vector( math.random() * self.worldDimensions.x, math.random() * self.worldDimensions.y).add(self.origin)

 
    # Tries self.candidateSeeds first, then samples using self.samplePoint
    def getSeed(self, major):
        # Candidate seeds first
        if (self.SEED_AT_ENDPOINTS and self.candidateSeeds(major).length > 0):
            while (self.candidateSeeds(major).length > 0):
                seed = self.candidateSeeds(major).pop()
                if (self.isValidSample(major, seed, self.paramsSq.dsep)): return seed

        seed = self.samplePoint()
        i = 0
        while (not self.isValidSample(major, seed, self.paramsSq.dsep)):
            if (i >= self.params.seedTries): return None
            seed = self.samplePoint()
            i += 1

        return seed

    def isValidSample(self, major, point, dSq, bothGrids=False):
        # dSq = dSq * point.distanceToSquared(Vector.zeroVector())
        gridValid = self.grid(major).isValidSample(point, dSq)
        if (bothGrids): gridValid = gridValid and self.grid(not major).isValidSample(point, dSq)
        return self.integrator.onLand(point) and gridValid

    def candidateSeeds(self, major):
        return self.candidateSeedsMajor if major else self.candidateSeedsMinor

    def streamlines(self, major):
        return self.streamlinesMajor if major else self.streamlinesMinor

    def grid(self, major):
        return self.majorGrid if major else self.minorGrid

    def pointInBounds(self, v):
        return (v.x >= self.origin.x and v.y >= self.origin.y and v.x < self.worldDimensions.x + self.origin.x and v.y < self.worldDimensions.y + self.origin.y)


    # Didn't end up using - bit expensive, used streamlineTurned instead
    # Stops spirals from forming
    # uses 0.5 dcirclejoin so that circles are still joined up
    # testSample is candidate to pushed on end of streamlineForwards
    # returns True if streamline collides with itself
    def doesStreamlineCollideSelf(self, testSample, streamlineForwards, streamlineBackwards):
        # Streamline long enough
        if (len(streamlineForwards) > self.nStreamlineLookBack):
            # Forwards check
            for i in range(len(streamlineForwards)):
                if (testSample.distanceToSquared(streamlineForwards[i]) < self.dcollideselfSq): return True
                i += self.nStreamlineStep

            # Backwards check
            for i in range(len(streamlineBackwards)):
                if (testSample.distanceToSquared(streamlineBackwards[i]) < self.dcollideselfSq): return True
                i += self.nStreamlineStep

        return False

    
    # Tests whether streamline has turned through greater than 180 degrees
    def streamlineTurned(seed, originalDir, point, direction):
        if (originalDir.dot(direction) < 0):
            # TODO optimise
            perpendicularVector = Vector(originalDir.y, -originalDir.x)
            isLeft = point.clone().sub(seed).dot(perpendicularVector) < 0
            directionUp = direction.dot(perpendicularVector) > 0
            return isLeft == directionUp
        return False


    # TODO self doesn't work well - consider something disallowing one direction (F/B) to turn more than 180 deg
    # One step of the streamline integration process
    def streamlineIntegrationStep(self, params, major, collideBoth):
        if (params.valid):
            params.streamline.push(params.previousPoint)
            nextDirection = self.integrator.integrate(params.previousPoint, major)

            # Stop at degenerate point
            if (nextDirection.lengthSq() < 0.01):
                params.valid = False
                return

            # Make sure we travel in the same direction
            if (nextDirection.dot(params.previousDirection) < 0): nextDirection.negate()

            nextPoint = params.previousPoint.clone().add(nextDirection)

            # Visualise stopping points
            # if (self.streamlineTurned(params.seed, params.originalDir, nextPoint, nextDirection)) {
                # params.valid = False
                # params.streamline.push(Vector.zeroVector())
            # }

            if (self.pointInBounds(nextPoint) and self.isValidSample(major, nextPoint, self.paramsSq.dtest, collideBoth) and not self.streamlineTurned(params.seed, params.originalDir, nextPoint, nextDirection)):
                params.previousPoint = nextPoint
                params.previousDirection = nextDirection
            else:
                # One more step
                params.streamline.push(nextPoint)
                params.valid = False

            pass

    # By simultaneously integrating in both directions we reduce the impact of circles not joining
    # up as the error matches at the join
    def integrateStreamline(self, seed, major):
        count = 0
        pointsEscaped = False  # True once two integration fronts have moved dlookahead away

        # Whether or not to test validity using both grid storages
        # (Collide with both major and minor)
        collideBoth = math.random() < self.params.collideEarly

        d = self.integrator.integrate(seed, major)

        forwardParams = StreamlineIntegration(seed, d, [seed], d, seed.clone().add(d), True)

        forwardParams.valid = self.pointInBounds(forwardParams.previousPoint)

        negD = d.clone().negate()
        backwardParams = StreamlineIntegration(seed, negD, [], negD, seed.clone().add(negD), True)

        backwardParams.valid = self.pointInBounds(backwardParams.previousPoint)

        while (count < self.params.pathIterations and (forwardParams.valid or backwardParams.valid)):
            self.streamlineIntegrationStep(forwardParams, major, collideBoth)
            self.streamlineIntegrationStep(backwardParams, major, collideBoth)

            # Join up circles
            sqDistanceBetweenPoints = forwardParams.previousPoint.distanceToSquared(backwardParams.previousPoint)

            if (not pointsEscaped and sqDistanceBetweenPoints > self.paramsSq.dcirclejoin): pointsEscaped = True

            if (pointsEscaped and sqDistanceBetweenPoints <= self.paramsSq.dcirclejoin):
                forwardParams.streamline.push(forwardParams.previousPoint)
                forwardParams.streamline.push(backwardParams.previousPoint)
                backwardParams.streamline.push(backwardParams.previousPoint)
                break
            count += 1

        backwardParams.streamline.reverse().push(*forwardParams.streamline)
        return backwardParams.streamline
