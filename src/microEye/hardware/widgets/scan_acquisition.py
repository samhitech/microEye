import os
from enum import Enum

import numpy as np
import pyqtgraph as pg
import qdarkstyle
import tifffile as tf
from pyqtgraph.parametertree import Parameter

from microEye.qt import (
    QApplication,
    Qt,
    QtWidgets,
    Signal,
    getExistingDirectory,
    getSaveFileName,
)
from microEye.utils import Tree
from microEye.utils.uImage import uImage


class TileImage:
    def __init__(self, uImage: uImage, index, position) -> None:
        self.uImage = uImage
        self.index = index
        self.position = position


class TiledImageSelector(QtWidgets.QWidget):
    positionSelected = Signal(float, float)

    def __init__(self, images: list[TileImage]) -> None:
        super().__init__()

        self.images = images

        central_layout = QtWidgets.QHBoxLayout()
        self.setLayout(central_layout)

        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.ci.setContentsMargins(0, 0, 0, 0)
        imageWidget.ci.setSpacing(0)
        imageWidget.sceneObj.sigMouseClicked.connect(self.clicked)

        for _idx, tImg in enumerate(images):
            vb: pg.ViewBox = imageWidget.addViewBox(*tImg.index)
            vb.setMouseEnabled(False, False)
            vb.setDefaultPadding(0.004)
            vb.setAspectLocked(True)
            vb.invertY()
            # menu: QMenu = vb.getMenu(None)
            # vb.action = QAction("Save Raw Data (.tif)")
            # menu.addAction(vb.action)
            # vb.action.triggered.connect(
            #     lambda: self.save_raw_data(tImg))
            img = pg.ImageItem(tImg.uImage._view)
            vb.addItem(img)
            vb.item = tImg

        self.imgView = pg.ImageView()
        self.setWindowTitle('Tiled Image Selector')

        central_layout.addWidget(imageWidget, 3)
        central_layout.addWidget(self.imgView, 4)

    def clicked(self, event):
        if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.LeftButton:
            self.save_raw_data(event.currentItem.item)
        elif (
            event.modifiers() == Qt.ControlModifier and event.button() == Qt.LeftButton
        ):
            self.save_raw_data_all()
        else:
            if event.double():
                self.positionSelected.emit(*event.currentItem.item.position)
            else:
                self.setWindowTitle(
                    'Tiled Image Selector ({}, {}) ({}, {})'.format(
                        *event.currentItem.item.index, *event.currentItem.item.position
                    )
                )
                self.imgView.setImage(event.currentItem.addedItems[0].image)

    def save_raw_data(self, img: TileImage):
        filename = None
        if filename is None:
            filename, _ = getSaveFileName(
                self, 'Save Raw Data', filter='Tiff Files (*.tif)'
            )

        if len(filename) > 0:
            tf.imwrite(filename, img.uImage.image, photometric='minisblack')

    def save_raw_data_all(self):
        directory = None
        if directory is None:
            directory = str(getExistingDirectory(self, 'Select Directory'))

        if len(directory) > 0:
            for idx, tImg in enumerate(self.images):
                tf.imwrite(
                    directory
                    + f'/{idx:03d}_image_y{tImg.index[0]:02d}_x{tImg.index[1]:02d}.tif',
                    tImg.uImage.image,
                    photometric='minisblack',
                )


class ScanParams(Enum):
    '''
    Enum class defining Scanning Acquisition parameters.
    '''

    DELAY = 'Delay [ms]'
    XY_SCAN = 'XY SCAN'
    Z_SCAN = 'Z SCAN'
    X_STEP = 'XY SCAN.X Steps'
    Y_STEP = 'XY SCAN.Y Steps'
    Z_STEP = 'Z SCAN.Z Steps'
    X_STEP_SIZE = 'XY SCAN.X Step Size [um]'
    Y_STEP_SIZE = 'XY SCAN.Y Step Size [um]'
    Z_STEP_SIZE = 'Z SCAN.Z Step Size [nm]'
    AVG_FRAMES = 'XY SCAN.Average [Frames]'

    XY_START = 'XY SCAN.Start XY-Scan'
    XY_LAST = 'XY SCAN.Last XY-Scan'
    XY_STOP = 'XY SCAN.Stop XY-Scan'

    N_FRAMES = 'Z SCAN.Frames per Z-Slice'
    Z_REVERSED = 'Z SCAN.Reversed'
    Z_DIRECTORY = 'Z SCAN.Save Directory'
    Z_START = 'Z SCAN.Start Z-Scan'
    Z_STOP = 'Z SCAN.Stop Z-Scan'
    Z_CAL = 'Z SCAN.Start Z-Calibration'

    def __str__(self):
        '''
        Return the last part of the enum value (Param name).
        '''
        return self.value.split('.')[-1]

    def get_path(self):
        '''
        Return the full parameter path.
        '''
        return self.value.split('.')


class ScanAcquisitionWidget(Tree):
    PARAMS = ScanParams
    startAcquisitionXY = Signal(tuple)
    stopAcquisitionXY = Signal()
    openLastTileXY = Signal()
    startAcquisitionZ = Signal(tuple)
    startCalibrationZ = Signal(tuple)
    stopAcquisitionZ = Signal()

    moveZ = Signal(bool, int)

    directoryChanged = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self._directory = os.path.join(os.path.expanduser('~'), 'Desktop')

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        This method creates and sets up the parameter tree structure for the
        `PzFocView` class.
        '''
        params = [
            {
                'name': str(ScanParams.DELAY),
                'type': 'int',
                'value': 200,
                'limits': [1, 1e4],
                'step': 50,
            },
            {
                'name': str(ScanParams.XY_SCAN),
                'type': 'group',
                'children': [
                    {
                        'name': str(ScanParams.X_STEP),
                        'type': 'int',
                        'value': 4,
                        'limits': [1, 100],
                        'step': 1,
                    },
                    {
                        'name': str(ScanParams.Y_STEP),
                        'type': 'int',
                        'value': 4,
                        'limits': [1, 100],
                        'step': 1,
                    },
                    {
                        'name': str(ScanParams.X_STEP_SIZE),
                        'type': 'float',
                        'value': 50.0,
                        'limits': [0.1, 500],
                        'step': 1,
                        'dec': False,
                        'decimals': 3,
                    },
                    {
                        'name': str(ScanParams.Y_STEP_SIZE),
                        'type': 'float',
                        'value': 50.0,
                        'limits': [0.1, 500],
                        'step': 1,
                        'dec': False,
                        'decimals': 3,
                    },
                    {
                        'name': str(ScanParams.AVG_FRAMES),
                        'type': 'int',
                        'value': 1,
                        'limits': [1, 128],
                        'step': 1,
                    },
                    {'name': str(ScanParams.XY_START), 'type': 'action'},
                    {'name': str(ScanParams.XY_STOP), 'type': 'action'},
                    {'name': str(ScanParams.XY_LAST), 'type': 'action'},
                ],
            },
            {
                'name': str(ScanParams.Z_SCAN),
                'type': 'group',
                'children': [
                    {
                        'name': str(ScanParams.Z_STEP),
                        'type': 'int',
                        'value': 5,
                        'limits': [2, 1e4],
                        'step': 1,
                    },
                    {
                        'name': str(ScanParams.Z_STEP_SIZE),
                        'type': 'int',
                        'value': 25,
                        'limits': [1, 2e4],
                        'step': 1,
                    },
                    {
                        'name': str(ScanParams.N_FRAMES),
                        'type': 'int',
                        'value': 10,
                        'limits': [1, 1e9],
                        'step': 1,
                    },
                    {
                        'name': str(ScanParams.Z_REVERSED),
                        'type': 'bool',
                        'value': False,
                    },
                    {
                        'name': str(ScanParams.Z_DIRECTORY),
                        'type': 'file',
                        'directory': os.path.join(os.path.expanduser('~'), 'Desktop'),
                        'value': os.path.join(os.path.expanduser('~'), 'Desktop'),
                        'fileMode': 'Directory',
                    },
                    {'name': str(ScanParams.Z_START), 'type': 'action'},
                    {'name': str(ScanParams.Z_STOP), 'type': 'action'},
                    {'name': str(ScanParams.Z_CAL), 'type': 'action'},
                ],
            },
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.get_param(ScanParams.XY_START).sigActivated.connect(
            lambda: self.startScanning(True)
        )
        self.get_param(ScanParams.XY_STOP).sigActivated.connect(
            lambda: self.stopAcquisitionXY.emit()
        )
        self.get_param(ScanParams.XY_LAST).sigActivated.connect(
            lambda: self.openLastTileXY.emit()
        )

        self.get_param(ScanParams.Z_START).sigActivated.connect(
            lambda: self.startScanning(False)
        )
        self.get_param(ScanParams.Z_STOP).sigActivated.connect(
            lambda: self.stopAcquisitionZ.emit()
        )
        self.get_param(ScanParams.Z_CAL).sigActivated.connect(
            lambda: self.startScanning(None)
        )

        self.get_param(ScanParams.Z_DIRECTORY).sigValueChanged.connect(
            self.directory_changed
        )

    def startScanning(self, XY: bool):
        '''
        Starts the scanning process based on the provided XY parameter.

        Parameters
        ----------
        XY : bool, optional
            If True, starts XY scanning with the parameters set in the
            GUI. If False, starts Z scanning with the parameters set in
            the GUI. If None, starts calibration for Z scanning with
            the parameters set in the GUI.

        '''
        if XY is None:
            self.startCalibrationZ.emit(self.get_params(False))
            return

        if XY:
            self.startAcquisitionXY.emit(self.get_params(XY))
        else:
            self.startAcquisitionZ.emit(self.get_params(XY))

    def directory_changed(self, param, value):
        '''Slot for browse clicked event'''
        self._directory = value
        self.directoryChanged.emit(self._directory)

    def get_params(self, XY: bool):
        '''
        Returns the scanning parameters for XY or Z scan.

        Parameters
        ----------
        XY : bool
            If True, returns parameters for XY scan. If False, returns
            parameters for Z scan.

        Returns
        -------
        tuple
            A tuple of scanning parameters depending on the value of XY.

            If XY is True, returns:
                - Number of X steps
                - Number of Y steps
                - X Step size (um)
                - Y Step size (um)
                - Delay (ms)
                - Average (frames)

            If XY is False, returns:
                - Number of Z steps
                - Z Step size (nm)
                - Delay (ms)
                - Number of Frames
                - Reversed (bool)
        '''
        if XY:
            return (
                self.get_param_value(ScanParams.X_STEP),
                self.get_param_value(ScanParams.Y_STEP),
                self.get_param_value(ScanParams.X_STEP_SIZE),
                self.get_param_value(ScanParams.Y_STEP_SIZE),
                self.get_param_value(ScanParams.DELAY),
                self.get_param_value(ScanParams.AVG_FRAMES),
            )
        else:
            return (
                self.get_param_value(ScanParams.Z_STEP),
                self.get_param_value(ScanParams.Z_STEP_SIZE),
                self.get_param_value(ScanParams.DELAY),
                self.get_param_value(ScanParams.N_FRAMES),
                self.get_param_value(ScanParams.Z_REVERSED),
            )

    def setActionsStatus(self, status: bool):
        '''
        Enables or disables the scanning actions based on the provided status parameter.

        Parameters
        ----------
        status : bool
            The status to set for the scanning actions.
            If True, the actions are enabled.
            If False, the actions are disabled.

        '''
        self.get_param(ScanParams.XY_START).setOpts(enabled=status)
        # self.get_param(ScanParams.XY_STOP).setOpts(enabled=status)
        self.get_param(ScanParams.XY_LAST).setOpts(enabled=status)
        self.get_param(ScanParams.Z_START).setOpts(enabled=status)
        # self.get_param(ScanParams.Z_STOP).setOpts(enabled=status)
        self.get_param(ScanParams.Z_CAL).setOpts(enabled=status)
        self.get_param(ScanParams.Z_DIRECTORY).setOpts(enabled=status)

    def __str__(self):
        return 'Scan Acquistion'

if __name__ == '__main__':
    x = np.linspace(255, 0, 256)
    y = np.linspace(0, 255, 256)
    z = np.concatenate([x, y])
    image = np.tile(z, (512, 1)).T

    data = []

    for i in range(10):
        for j in range(10):
            uImg = uImage(image)
            uImg.equalizeLUT()
            tImg = TileImage(uImg, [i, j], [17, 17])
            data.append(tImg)

    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    win = TiledImageSelector(data)
    win.positionSelected.connect(lambda x, y: print(x, y))
    win.show()

    app.exec()
