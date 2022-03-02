import Vector
import math
# import * as PolyK from 'polyk'
# import * as jsts from 'jsts'

class PolygonUtil:
    # private static geometryFactory =  jsts.geom.GeometryFactory()

    # Slices rectangle by line, returning smallest polygon
    @staticmethod
    def sliceRectangle(origin, worldDimensions, p1, p2):
        rectangle = [
            origin.x, origin.y,
            origin.x + worldDimensions.x, origin.y,
            origin.x + worldDimensions.x, origin.y + worldDimensions.y,
            origin.x, origin.y + worldDimensions.y,
        ]
        # sliced = PolyK.Slice(rectangle, p1.x, p1.y, p2.x, p2.y).map(p => PolygonUtil.polygonArrayToPolygon(p))
        # minArea = PolygonUtil.calcPolygonArea(sliced[0])

        # if (sliced.length > 1 and PolygonUtil.calcPolygonArea(sliced[1]) < minArea) {
        #     return sliced[1]
        # }

        # return sliced[0]
        pass

    # Used to create sea polygon
    @staticmethod
    def lineRectanglePolygonIntersection(origin, worldDimensions, line):
        # jstsLine = PolygonUtil.lineToJts(line)
        bounds = [
            origin,
             Vector(origin.x + worldDimensions.x, origin.y),
             Vector(origin.x + worldDimensions.x, origin.y + worldDimensions.y),
             Vector(origin.x, origin.y + worldDimensions.y),
        ]
        # boundingPoly = PolygonUtil.polygonToJts(bounds)
        # union = boundingPoly.getExteriorRing().union(jstsLine)
        # polygonizer =  (jsts.operation as any).polygonize.Polygonizer()
        # polygonizer.add(union)
        # polygons = polygonizer.getPolygons()

        # let smallestArea = Infinity
        # let smallestPoly
        # for (let i = polygons.iterator() i.hasNext()) {
        #     polygon = i.next()
        #     area = polygon.getArea()
        #     if (area < smallestArea) {
        #         smallestArea = area
        #         smallestPoly = polygon
        #     }
        # }

        # if (!smallestPoly) return []
        # return smallestPoly.getCoordinates().map((c: any) =>  Vector(c.x, c.y))
        pass

    @staticmethod
    def calcPolygonArea(polygon):
        total = 0
        for i in range(len(polygon)):
            addX = polygon[i].x
            addY = polygon[0 if i == len(polygon)-1 else i+1].y
            subX = polygon[0 if i == len(polygon)-1 else i+1].x
            subY = polygon[i].y

            total += (addX * addY * 0.5)
            total -= (subX * subY * 0.5)

        return math.abs(total)

    # Recursively divide a polygon by its longest side until the minArea stopping condition is met
    @staticmethod
    def subdividePolygon(p, minArea):
        area = PolygonUtil.calcPolygonArea(p)
        if (area < 0.5 * minArea): return []

        divided = []  # Array of polygons
        longestSideLength = 0
        longestSide = [p[0], p[1]]
        perimeter = 0
        for i in range(len(p)):
            sideLength = p[i].clone().sub(p[(i+1) % p.length]).length()
            perimeter += sideLength
            if (sideLength > longestSideLength):
                longestSideLength = sideLength
                longestSide = [p[i], p[(i+1) % p.length]]


        # Shape index
        # Using rectangle ratio of 1:4 as limit
        if (area / perimeter * perimeter < 0.04): return []

        if (area < 2 * minArea): return [p]

        # Between 0.4 and 0.6
        deviation = (math.random() * 0.2) + 0.4

        averagePoint = longestSide[0].clone().add(longestSide[1]).multiplyScalar(deviation)
        differenceVector = longestSide[0].clone().sub(longestSide[1])
        perpVector = ( Vector(differenceVector.y, -1 * differenceVector.x)).normalize().multiplyScalar(100)

        bisect = [averagePoint.clone().add(perpVector), averagePoint.clone().sub(perpVector)]

        # Array of polygons
        # TODO
        # try {
        #     sliced = PolyK.Slice(PolygonUtil.polygonToPolygonArray(p), bisect[0].x, bisect[0].y, bisect[1].x, bisect[1].y)
        #     // Recursive call
        #     for (s of sliced) {
        #         divided.push(...PolygonUtil.subdividePolygon(PolygonUtil.polygonArrayToPolygon(s), minArea))
        #     }

        #     return divided
        # } catch (error) {
        #     log.error(error)
        #     return []
        # }

#     /**
#      * Shrink or expand polygon
#      */
#     public static resizeGeometry(geometry: Vector[], spacing: number, isPolygon=true): Vector[] {
#         try {
#             jstsGeometry = isPolygon? PolygonUtil.polygonToJts(geometry) : PolygonUtil.lineToJts(geometry)
#             resized = jstsGeometry.buffer(spacing, undefined, (jsts as any).operation.buffer.BufferParameters.CAP_FLAT)
#             if (!resized.isSimple()) {
#                 return []
#             }
#             return resized.getCoordinates().map(c =>  Vector(c.x, c.y))
#         } catch (error) {
#             log.error(error)
#             return []
#         }
#     }

#     public static averagePoint(polygon: Vector[]): Vector {
#         if (polygon.length === 0) return Vector.zeroVector()
#         sum = Vector.zeroVector()
#         for (v of polygon) {
#             sum.add(v)
#         }
#         return sum.divideScalar(polygon.length)
#     }

#     public static insidePolygon(point: Vector, polygon: Vector[]): boolean {
#         // ray-casting algorithm based on
#         // http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html

#         if (polygon.length === 0) {
#             return false
#         }

#         let inside = false
#         for (let i = 0, j = polygon.length - 1 i < polygon.length j = i++) {
#             xi = polygon[i].x, yi = polygon[i].y
#             xj = polygon[j].x, yj = polygon[j].y

#             intersect = ((yi > point.y) != (yj > point.y))
#                 and (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)
#             if (intersect) inside = !inside
#         }

#         return inside
#     }

#     public static pointInRectangle(point: Vector, origin: Vector, dimensions: Vector): boolean {
#         return point.x >= origin.x and point.y >= origin.y and point.x <= dimensions.x and point.y <= dimensions.y
#     }

#     private static lineToJts(line: Vector[]): jsts.geom.LineString {
#         coords = line.map(v =>  jsts.geom.Coordinate(v.x, v.y))
#         return PolygonUtil.geometryFactory.createLineString(coords)
#     }

#     private static polygonToJts(polygon: Vector[]): jsts.geom.Polygon {
#         geoInput = polygon.map(v =>  jsts.geom.Coordinate(v.x, v.y))
#         geoInput.push(geoInput[0])  // Create loop
#         return PolygonUtil.geometryFactory.createPolygon(PolygonUtil.geometryFactory.createLinearRing(geoInput), [])
#     }

    # [ v.x, v.y, v.x, v.y ]...
    def polygonArrayToPolygon(p):
        outP = []
        for v in p:
            outP.push(v.x)
            outP.push(v.y)
        return outP


    # [ v.x, v.y, v.x, v.y ]...
    def polygonArrayToPolygon(p):
        outP = []
        for i in range(len(p)/2):
            outP.push( Vector(p[2*i], p[2*i + 1]))
        return outP
