# 1. Main
#   a. Tensor Field GUI



# Input

import Vector
import tensor_field

def growthRule():
    # TODO: Get growthRule map Image (PYTHON-wala)
    # TODO: Find centers for each segment in a dictionary
    rule = {
        'Organic': [ {'centre':[0,0], 'size': int} ],
        'Grid': [ {'centre':[0,0], 'size': int} ],
        'Radial': [ {'centre':[0,0], 'size': int} ],
    }
    return rule

def 

def generate():
    # TODO: window size: width and height
    # TODO: zooming i.e. worldDimensions wrt windowSize

    # TODO: locations based on growthRule map (PYTHON-wala)
    x,y = 0,0
    addGridAtLocation(x,y)

    # this.addGridAtLocation(newOrigin);
    # this.addGridAtLocation(newOrigin.clone().add(size));
    # this.addGridAtLocation(newOrigin.clone().add(new Vector(size.x, 0)));
    # this.addGridAtLocation(newOrigin.clone().add(new Vector(0, size.y)));
    # this.addRadialRandom();



def addGridAtLocation(location: Vector):
    addGrid(location, size, decay, angle )
    # (width / 4, width) // Size
    # 50 // Decay
    # (Math.PI / 2)


def addRadialRandom():
        const width = this.domainController.worldDimensions.x;
        this.addRadial(this.randomLocation(),
            Util.randomRange(width / 10, width / 5),  // Size
            Util.randomRange(50));  // Decay
    }

def addGridRandom(): void {
        this.addGridAtLocation(this.randomLocation());
    }

    private addGridAtLocation(location: Vector): void {
        const width = this.domainController.worldDimensions.x;
        this.
    }

    /**
     * World-space random location for tensor field spawn
     * Sampled from middle of screen (shrunk rectangle)
     */
    private randomLocation(): Vector {
        const size = this.domainController.worldDimensions.multiplyScalar(this.TENSOR_SPAWN_SCALE);
        const location = new Vector(Math.random(), Math.random()).multiply(size);
        const newOrigin = this.domainController.worldDimensions.multiplyScalar((1 - this.TENSOR_SPAWN_SCALE) / 2);
        return location.add(this.domainController.origin).add(newOrigin);
    }

    private getCrossLocations(): Vector[] {
        // Gets grid of points for vector field vis in world space
        const diameter = this.TENSOR_LINE_DIAMETER / this.domainController.zoom;
        const worldDimensions = this.domainController.worldDimensions;
        const nHor = Math.ceil(worldDimensions.x / diameter) + 1; // Prevent pop-in
        const nVer = Math.ceil(worldDimensions.y / diameter) + 1;
        const originX = diameter * Math.floor(this.domainController.origin.x / diameter);
        const originY = diameter * Math.floor(this.domainController.origin.y / diameter);

        const out = [];
        for (let x = 0; x <= nHor; x++) {
            for (let y = 0; y <= nVer; y++) {
                out.push(new Vector(originX + (x * diameter), originY + (y * diameter)));
            }
        }

        return out;
    }
