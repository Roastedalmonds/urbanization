import Vector
import Tensor
import math
from abc import abstractmethod, abstractproperty

# Grid or Radial field to be combined with others to create the tensor field
class BasisField:
    # TODO @abstractproperty FOLDER_NAME, FIELD_TYPE, folderNameIndex, parentFolder, folder, _centre
    # abstract readonly FOLDER_NAME: string
    # abstract readonly FIELD_TYPE: number
    # protected static folderNameIndex: number = 0
    # protected parentFolder: dat.GUI
    # protected folder: dat.GUI
    # protected _centre: Vector
    
    def __init__(self, centre, _size, _decay):
        # constructor(centre: Vector, protected _size: number, protected _decay: number)
        self._centre = centre.clone()

    def set_centre(self, centre): self._centre.copy(centre)

    def get_centre(self): return self._centre.clone()

    def set_decay(self, decay): self._decay = decay

    def set_size(self, size): self._size = size

    def dragStartListener(self):
        self.setFolder()
        pass

    def dragMoveListener(self, delta):
        # Delta assumed to be in world space (only relevant when zoomed)
        self._centre.add(delta)
        pass

    def getWeightedTensor(self, point, smooth): return self.getTensor(point).scale(self.getTensorWeight(point, smooth))

    # def setFolder(self):
    #     if (self.parentFolder.__folders):
    #         for (folderName in self.parentFolder.__folders):
    #             self.parentFolder.__folders[folderName].close()
    #         self.folder.open()
    #     pass

    # abstract getTensor(point: Vector): Tensor


    def removeFolderFromParent(self):
        if (self.parentFolder.__folders and Object.values(self.parentFolder.__folders).indexOf(self.folder) >= 0):
            self.parentFolder.removeFolder(self.folder)

    # Creates a folder and adds it to the GUI to control params
    def setGui(self, parent: dat.GUI, folder: dat.GUI):
        self.parentFolder = parent
        self.folder = folder
        folder.add(self._centre, 'x')
        folder.add(self._centre, 'y')
        folder.add(self, '_size')
        folder.add(self, '_decay', -50, 50)
        pass

    # Interpolates between (0 and 1)^decay
    def __getTensorWeight(self, point, smooth):
        normDistanceToCentre = point.clone().sub(self._centre).length() / self._size
        if (smooth): return normDistanceToCentre ** -self._decay

        # Stop (** 0) turning weight into 1, filling screen even when outside 'size'
        if (self._decay == 0 and normDistanceToCentre >= 1): return 0
        return math.max(0, (1 - normDistanceToCentre)) ** self._decay


class Grid(BasisField):
    # TODO
    FOLDER_NAME, FIELD_TYPE = 'Grid #TODO FolderNameIndex', FIELD_TYPE[0]
    # readonly FOLDER_NAME = `Grid ${Grid.folderNameIndex++}`
    # readonly FIELD_TYPE = FIELD_TYPE.Grid

    def __init__(self, centre, size, decay, _theta):
        # constructor(centre: Vector, size: number, decay: number, private _theta: number)
        super().__init__(centre, size, decay)

    def set_theta(self, theta): self._theta = theta

    def getTensor(self, point):
        cos = math.cos(2 * self._theta)
        sin = math.sin(2 * self._theta)
        return Tensor(1, [cos, sin])

    def setGui(parent: dat.GUI, folder: dat.GUI):
        # super.setGui(parent, folder)

        # GUI in degrees, convert to rads
        # thetaProp = {theta: self._theta * 180 / math.PI}
        # thetaController = folder.add(thetaProp, 'theta', -90, 90)
        # thetaController.onChange(theta => self._theta = theta * (math.PI / 180))
        pass


class Radial(BasisField):
    # TODO
    # @property
    FOLDER_NAME, FIELD_TYPE = 'Radial #TODO FolderNameIndex', FIELD_TYPE[1]
    # readonly FOLDER_NAME = `Radial ${Radial.folderNameIndex++}`
    # readonly FIELD_TYPE = FIELD_TYPE.Radial

    def __init__(self, centre, size, decay):
        # constructor(centre: Vector, size: number, decay: number)
        super().__init__(centre, size, decay)

    def getTensor(self, point):
        t = point.clone().sub(self._centre)
        t1 = t.y**2 - t.x**2
        t2 = -2 * t.x * t.y
        return Tensor(1, [t1, t2])

FIELD_TYPE = [Grid, Radial]