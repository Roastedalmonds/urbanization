# 1. Main
#   a. Tensor Field GUI



# Input

import math
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

def addLayout(rule):
    # for each in rule dict:
    # Add grid
    tensor_field.addGrid(rule['Grid'][0]['centre'], rule['Grid'][0]['size'], 50, math.PI/2)
    # randomrange(width / 4, width) // Size
    # randomrange(50) // Decay
    # randomrange(Math.PI / 2) // theta


    # Add organic


    # Add Radial
    tensor_field.addGrid(rule['Grid'][0]['centre'], rule['Grid'][0]['size'], 50, math.PI/2)
    pass

def generate():
    # TODO: window size: width and height
    # TODO: zooming i.e. worldDimensions wrt windowSize

    # TODO: locations based on growthRule map (PYTHON-wala)
    rule = growthRule()
    addLayout(rule)



    # private getCrossLocations(): Vector[] {
    #     // Gets grid of points for vector field vis in world space
    #     const diameter = this.TENSOR_LINE_DIAMETER / this.domainController.zoom;
    #     const worldDimensions = this.domainController.worldDimensions;
    #     const nHor = Math.ceil(worldDimensions.x / diameter) + 1; // Prevent pop-in
    #     const nVer = Math.ceil(worldDimensions.y / diameter) + 1;
    #     const originX = diameter * Math.floor(this.domainController.origin.x / diameter);
    #     const originY = diameter * Math.floor(this.domainController.origin.y / diameter);

    #     const out = [];
    #     for (let x = 0; x <= nHor; x++) {
    #         for (let y = 0; y <= nVer; y++) {
    #             out.push(new Vector(originX + (x * diameter), originY + (y * diameter)));
    #         }
    #     }

    #     return out;
    # }
