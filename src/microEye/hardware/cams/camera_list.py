import os
import typing

from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.micam import miCamera, miDummy
from microEye.qt import QtCore, QtGui, QtWidgets, Signal
from microEye.utils.thread_worker import QThreadWorker

try:
    from pyueye import ueye

    from microEye.hardware.cams.ueye.ueye_camera import IDS_Camera
    from microEye.hardware.cams.ueye.ueye_panel import IDS_Panel
except Exception:
    ueye = None
    IDS_Camera = None
    IDS_Panel = None

from microEye.hardware.cams.vimba import INSTANCE, vb

if vb is not None:
    from microEye.hardware.cams.vimba.vimba_cam import vimba_cam
    from microEye.hardware.cams.vimba.vimba_panel import Vimba_Panel
else:
    vimba_cam = None
    Vimba_Panel = None

from microEye.hardware.cams.basler.basler_cam import basler_cam
from microEye.hardware.cams.basler.basler_panel import Basler_Panel
from microEye.hardware.cams.dummy.dummy_panel import Dummy_Panel
from microEye.hardware.cams.pycromanager import PycroCamera
from microEye.hardware.cams.pycromanager.pycro_panel import PycroPanel
from microEye.hardware.cams.thorlabs.thorlabs import thorlabs_camera
from microEye.hardware.cams.thorlabs.thorlabs_panel import Thorlabs_Panel

try:
    CAMERA_CLASSES = {cls.__name__: cls for cls in miCamera.__subclasses__()}
except Exception:
    CAMERA_CLASSES = {}
    import traceback

    traceback.print_exc()

CAMERA_CONFIGS = {
    'Basler': {
        'driver': 'Basler',
        'camera_class': basler_cam,
        'camera_args': ['FullName'],
        'panel_class': Basler_Panel,
    },
    'miDummy': {
        'driver': 'miDummy',
        'camera_class': miDummy,
        'camera_args': None,
        'panel_class': Dummy_Panel,
    },
    'PycroCore': {
        'driver': 'PycroCore',
        'camera_class': PycroCamera,
        'camera_args': ['Camera ID', 'Port'],
        'panel_class': PycroPanel,
    },
    'UC480': {
        'driver': 'UC480',
        'camera_class': thorlabs_camera,
        'camera_args': ['Camera ID'],
        'panel_class': Thorlabs_Panel,
    },
    'uEye': {
        'driver': 'uEye',
        'camera_class': IDS_Camera,
        'camera_args': ['Camera ID'],
        'panel_class': IDS_Panel,
    },
    'Vimba': {
        'driver': 'Vimba',
        'camera_class': vimba_cam,
        'camera_args': ['Camera ID'],
        'panel_class': Vimba_Panel,
    },
}

class CameraList(QtWidgets.QWidget):
    '''
    A widget for displaying and managing a list of cameras.
    '''

    cameraAdded = Signal(Camera_Panel, bool)
    cameraRemoved = Signal(Camera_Panel, bool)

    CAMERAS : dict[str, list[dict]] = {
        'Basler': [],
        'miDummy': [],
        'PycroCore': [],
        'UC480': [],
        'uEye': [],
        'Vimba': [],
    }

    def __init__(self, parent: typing.Optional['QtWidgets.QWidget'] = None):
        '''
        Initialize the camera list widget.

        Parameters
        ----------
        parent : typing.Optional[QtWidgets.QWidget], optional
            The parent widget of this camera list widget.
        '''
        super().__init__(parent=parent)

        self.cam_list = None
        self.item_model = QtGui.QStandardItemModel()
        self.cached_autofocusCam = None

        #  Layout
        self.InitLayout()

    def InitLayout(self):
        '''
        Initialize the layout of the widget.
        '''

        # main layout
        self.mainLayout = QtWidgets.QVBoxLayout()

        self.cam_table = QtWidgets.QTableView()
        self.cam_table.setModel(self.item_model)
        self.cam_table.clearSelection()
        self.cam_table.horizontalHeader().setStretchLastSection(True)
        self.cam_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.cam_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.cam_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )

        self.HL_buttons = QtWidgets.QHBoxLayout()

        self.add_cam = QtWidgets.QPushButton(
            'Add Camera', clicked=lambda: self.add_camera_clicked()
        )

        self.remove_cam = QtWidgets.QPushButton(
            'Remove Camera', clicked=lambda: self.remove_camera_clicked()
        )

        self.refresh = QtWidgets.QPushButton(
            'Refresh List', clicked=lambda: self.refresh_list()
        )

        self.HL_buttons.addWidget(self.add_cam)
        self.HL_buttons.addWidget(self.remove_cam)
        self.HL_buttons.addWidget(self.refresh)

        self.mainLayout.addWidget(self.cam_table)
        self.mainLayout.addLayout(self.HL_buttons)

        self.setLayout(self.mainLayout)

        self.refresh_list()

    @property
    def autofocusCam(self) -> typing.Union[Camera_Panel, None]:
        '''
        Get the autofocus camera panel.

        Returns
        -------
        Camera_Panel | None
            The autofocus camera panel, or None if no autofocus camera is available.
        '''
        if self.cached_autofocusCam is None:
            self.cached_autofocusCam = next(
                (
                    cam['Panel']
                    for _, cam_list in CameraList.CAMERAS.items()
                    for cam in cam_list
                    if cam['IR']
                ),
                None,
            )
        return self.cached_autofocusCam

    def add_camera_clicked(self):
        '''
        Add a camera when the "Add Camera" button is clicked.
        '''
        if len(self.cam_table.selectedIndexes()) > 0:
            cam = self.cam_list[self.cam_table.currentIndex().row()]

            # create a dialog with radio buttons
            dialog, ok = QtWidgets.QInputDialog.getItem(
                self,
                'Add Camera',
                'Choose camera type:',
                ('Acquisition', 'Autofocus IR'),
            )

            if ok and dialog is not None:
                if dialog == 'Acquisition':
                    panel = self.add_camera(cam)
                    if panel:
                        self.cameraAdded.emit(panel, False)
                elif dialog == 'Autofocus IR':
                    if self.autofocusCam is None:
                        panel = self.add_camera(cam, True)
                        if panel:
                            self.cameraAdded.emit(panel, True)
                    else:
                        self._display_warning_message(
                            'Autofocus IR camera has already been added!'
                        )

            self.refresh_list()
        else:
            self._display_warning_message('Please select a device.')

    def add_camera(self, cam, mini=False):
        '''
        Add a camera.

        Parameters
        ----------
        cam : dict
            The camera information dictionary.
        mini : bool, optional
            True to add a mini camera panel, False to add a full camera panel.

        Returns
        -------
        Camera_Panel | None
            The camera panel, or None if the camera could not be added.
        '''
        if cam['InUse'] == 0:
            driver = cam['Driver']
            config = CAMERA_CONFIGS.get(driver)
            if config:
                return self._add_camera_generic(cam, mini, config)
            else:
                self._display_warning_message(f'Unsupported camera driver: {driver}')
        else:
            self._display_warning_message('Device is in use or already added.')


    def _add_camera_generic(self, cam, mini, config):
        '''
        Generic camera/panel adder.

        Parameters
        ----------
        cam : dict
            The camera information dictionary.
        mini : bool
            True to add a mini camera panel, False to add a full camera panel.
        config : dict
            Configuration for camera type (see below).

        Returns
        -------
        Camera_Panel | None
            The camera panel, or None if the camera could not be added.
        '''
        driver = config['driver']
        camera_class = config['camera_class']
        camera_args = config['camera_args']
        panel_class = config['panel_class']

        args = [mini, self.get_cam_title(cam)]

        if camera_args:
            camera = camera_class(*[cam[arg] for arg in camera_args])
            args.insert(0, camera)

        panel : Camera_Panel = panel_class(*args)

        # Add to CAMERAS dict
        CameraList.CAMERAS[driver].append(
            {'Camera': panel.cam, 'Panel': panel, 'IR': mini, 'Info': cam}
        )
        return panel

    def _display_warning_message(self, message):
        '''
        Display a warning message.

        Parameters
        ----------
        message : str
            The warning message to display.
        '''
        QtWidgets.QMessageBox.warning(
            self, 'Warning', message, QtWidgets.QMessageBox.StandardButton.Ok
        )

    def get_cam_title(self, cam: dict):
        '''
        Get the camera title.

        Parameters
        ----------
        cam : dict
            The camera information dictionary.

        Returns
        -------
        str
            The camera title.
        '''
        return f"{cam['Driver']} | {cam['Model']}"

    def remove_camera_clicked(self):
        '''
        Remove a camera when the "Remove Camera" button is clicked.
        '''
        if len(self.cam_table.selectedIndexes()) > 0:
            cam = self.cam_list[self.cam_table.currentIndex().row()]

            # Display a confirmation dialog
            confirm = QtWidgets.QMessageBox.question(
                self,
                'Confirmation',
                f'Do you want to remove this camera {self.get_cam_title(cam)}?',
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )

            if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
                self.remove_camera(cam)

            self.refresh_list()
        else:
            self._display_warning_message('Please select a device.')

    def remove_camera(self, cam):
        '''
        Remove a camera.

        Parameters
        ----------
        cam : dict
            The camera information dictionary.
        '''
        cams: list[dict] = CameraList.CAMERAS.get(cam['Driver'], [])
        if cams:
            for item in cams:
                pan: Camera_Panel = item['Panel']
                if pan.title() == self.get_cam_title(cam):
                    if pan.cam.acquisition:
                        self._display_warning_message(
                            'Please stop acquisition before removing!'
                        )
                    else:
                        pan.dispose()

                        pan._dispose_cam = True
                        if pan.acq_job is not None:
                            pan.acq_job.stop_threads = True
                            pan.acq_job.c_event.set()

                        cams.remove(item)
                        if item['IR']:
                            pan.setParent(None)
                        else:
                            pan.close()
                            pan.setParent(None)
                        self.cameraRemoved.emit(pan, item['IR'])
                    return
        else:
            self._display_warning_message('Device/Panel not found!')

    def removeAllCameras(self):
        '''
        Remove all cameras.
        Stops any active acquisitions and properly disposes of camera resources.
        '''
        for _, camera_list in self.CAMERAS.items():
            # Create a copy of the list since we'll be modifying it during iteration
            for camera_info in camera_list[:]:
                panel: Camera_Panel = camera_info['Panel']

                # Stop acquisition if running
                if panel.cam.acquisition:
                    panel.stop()

                # Clean up based on camera type
                panel.dispose()

                # Stop any acquisition jobs
                if panel.acq_job is not None:
                    panel.acq_job.stop_threads = True
                    panel.acq_job.c_event.set()

                # Remove from parent and clean up panel
                panel._dispose_cam = True
                if camera_info['IR']:
                    panel.setParent(None)
                else:
                    panel.close()
                    panel.setParent(None)

                # Remove from camera list
                camera_list.remove(camera_info)

    def refresh_list(self):
        '''
        Refresh the camera list.
        '''
        # import faulthandler
        # faulthandler.enable()

        self.refresh.setEnabled(False)

        def _refresh_list(event):
            cam_list = []
            for key, camera_class in CAMERA_CLASSES.items():
                if 'thorlabs' in key.lower() and not os.path.exists(
                    thorlabs_camera.uc480_file
                ):
                    continue
                try:
                    if camera_class:
                        cam_list += camera_class.get_camera_list()
                except Exception as e:
                    print(f'Error getting camera list for {key}: {e}')

            return cam_list

        def done(result):
            self.cam_list = result
            self.update_list()
            self.refresh.setEnabled(True)

        # worker = QThreadWorker(_refresh_list)
        # worker.signals.result.connect(done)

        # QtCore.QThreadPool.globalInstance().start(worker)
        done(_refresh_list(None))

    def update_list(self):
        self.item_model = QtGui.QStandardItemModel(len(self.cam_list), 8)

        self.item_model.setHorizontalHeaderLabels(
            [
                'In Use',
                'Camera ID',
                'Device ID',
                'Model',
                'Serial',
                'Status',
                'Sensor ID',
                'Driver',
            ]
        )

        for i in range(len(self.cam_list)):
            self.item_model.setItem(
                i, 0, QtGui.QStandardItem(str(self.cam_list[i]['InUse']))
            )
            self.item_model.setItem(
                i, 1, QtGui.QStandardItem(str(self.cam_list[i]['Camera ID']))
            )
            self.item_model.setItem(
                i, 2, QtGui.QStandardItem(str(self.cam_list[i]['Device ID']))
            )
            self.item_model.setItem(
                i, 3, QtGui.QStandardItem(self.cam_list[i]['Model'])
            )
            self.item_model.setItem(
                i, 4, QtGui.QStandardItem(self.cam_list[i]['Serial'])
            )
            self.item_model.setItem(
                i, 5, QtGui.QStandardItem(str(self.cam_list[i]['Status']))
            )
            self.item_model.setItem(
                i, 6, QtGui.QStandardItem(str(self.cam_list[i]['Sensor ID']))
            )
            self.item_model.setItem(
                i, 7, QtGui.QStandardItem(str(self.cam_list[i]['Driver']))
            )

        self.cam_table.setModel(self.item_model)

        # Update the cached_autofocusCam value
        self.cached_autofocusCam = None

    def snap_image(self):
        '''
        Snap an image on all non IR cameras.
        '''
        for _, cam_list in CameraList.CAMERAS.items():
            for cam in cam_list:
                if not cam['IR']:
                    cam['Panel'].capture_image()

    def get_config(self):
        '''
        Get the configuration of all added cameras.

        This allows adding the same cameras again later.

        Returns
        -------
        dict
            A dictionary containing the configuration of all added cameras.
        '''
        config : list[dict] = []

        for _, cam_list in CameraList.CAMERAS.items():
            for cam in cam_list:
                cam_config : dict = cam['Info'].copy()
                cam_config['IR'] = cam['IR']
                config.append(cam_config)
        return config

    def load_config(self, config: list[dict]):
        '''
        Adds the cameras in the given configuration.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration of all cameras to add.
        '''
        for cam in config:
            is_IR = cam.pop('IR', False)
            # check if camera is available in the current list
            if cam not in self.cam_list:
                print(f'Camera {self.get_cam_title(cam)} not available; skipping.')
                continue
            if self.autofocusCam is not None and is_IR:
                print('Skipping adding autofocus IR camera; one already exists.')
                continue
            panel = self.add_camera(cam, is_IR)
            if panel:
                self.cameraAdded.emit(panel, is_IR)
        self.refresh_list()
