# from pyrsistent import field
from tensor_field import TensorField
import Vector
from streamlines import StreamlineParams

class FieldIntegrator:
    def __init__(self):
        # constructor(protected field: TensorField) {}
        pass

    def sampleFieldVector(self, point, major):
        tensor = self.field.samplePoint(point)
        if (major): return tensor.getMajor()
        return tensor.getMinor()

    def onLand(self, point): return self.field.onLand(point)

class EulerIntegrator(FieldIntegrator):
    def __init__(self, field, params):
        super().__init__(field)

    def integrate(self, point, major):
        return self.sampleFieldVector(point, major).multiplyScalar(self.params.dstep)

class RK4Integrator(FieldIntegrator):
    def __init__(self, field, params):
        super().__init__(field)

    def integrate(self, point, major):
        k1 = self.sampleFieldVector(point, major)
        k23 = self.sampleFieldVector(point.clone().add(Vector.fromScalar(self.params.dstep / 2)), major)
        k4 = self.sampleFieldVector(point.clone().add(Vector.fromScalar(self.params.dstep)), major)

        return k1.add(k23.multiplyScalar(4)).add(k4).multiplyScalar(self.params.dstep / 6)