import os
import typing

from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.dummy.dummy_panel import Dummy_Panel
from microEye.hardware.cams.micam import miDummy
from microEye.hardware.cams.thorlabs.thorlabs import CMD, thorlabs_camera
from microEye.hardware.cams.thorlabs.thorlabs_panel import Thorlabs_Panel
from microEye.qt import QtGui, QtWidgets, Signal

try:
    from pyueye import ueye

    from microEye.hardware.cams.ueye.ueye_camera import IDS_Camera
    from microEye.hardware.cams.ueye.ueye_panel import IDS_Panel
except Exception:
    ueye = None
    IDS_Camera = None

try:
    import vimba as vb

    from microEye.hardware.cams.vimba.vimba_cam import get_camera_list, vimba_cam
    from microEye.hardware.cams.vimba.vimba_panel import Vimba_Panel
except Exception:
    vb = None

    def get_camera_list():
        return []


class CameraList(QtWidgets.QWidget):
    '''
    A widget for displaying and managing a list of cameras.
    '''

    cameraAdded = Signal(Camera_Panel, bool)
    cameraRemoved = Signal(Camera_Panel, bool)

    CAMERAS = {'uEye': [], 'Vimba': [], 'UC480': [], 'miDummy': []}

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
            if driver == 'uEye':
                return self._add_ids_camera(cam, cam['Camera ID'], mini)
            elif driver == 'UC480':
                return self._add_thorlabs_camera(cam, cam['Camera ID'], mini)
            elif driver == 'Vimba':
                return self._add_vimba_camera(cam, cam['Camera ID'], mini)
            elif driver == 'miDummy':
                return self._add_dummy_camera(cam, mini)
            else:
                self._display_warning_message(f'Unsupported camera driver: {driver}')
        else:
            self._display_warning_message('Device is in use or already added.')

    def _add_ids_camera(self, cam, camera_id, mini):
        '''
        Add an IDS camera.

        Parameters
        ----------
        cam : dict
            The camera information dictionary.
        camera_id : int
            The camera ID.
        mini : bool
            True to add a mini camera panel, False to add a full camera panel.

        Returns
        -------
        Camera_Panel | None
            The camera panel, or None if the camera could not be added.
        '''
        ids_cam = IDS_Camera(camera_id)
        ids_cam.initialize()
        ids_panel = IDS_Panel(ids_cam, mini, self.get_cam_title(cam))
        CameraList.CAMERAS['uEye'].append(
            {'Camera': ids_cam, 'Panel': ids_panel, 'IR': mini}
        )
        return ids_panel

    def _add_thorlabs_camera(self, cam, camera_id, mini):
        '''
        Add a Thorlabs camera.

        Parameters
        ----------
        cam : dict
            The camera information dictionary.
        camera_id : int
            The camera ID.
        mini : bool
            True to add a mini camera panel, False to add a full camera panel.

        Returns
        -------
        Camera_Panel | None
            The camera panel, or None if the camera could not be added.
        '''
        thor_cam = thorlabs_camera(camera_id)
        n_ret = thor_cam.initialize()
        if n_ret == CMD.IS_SUCCESS:
            thor_panel = Thorlabs_Panel(thor_cam, mini, self.get_cam_title(cam))
            CameraList.CAMERAS['UC480'].append(
                {'Camera': thor_cam, 'Panel': thor_panel, 'IR': mini}
            )
            return thor_panel
        else:
            self._display_warning_message('Thorlabs camera initialization failed.')

    def _add_vimba_camera(self, cam, camera_id, mini):
        '''
        Add a Vimba camera.

        Parameters
        ----------
        cam : dict
            The camera information dictionary.
        camera_id : int
            The camera ID.
        mini : bool
            True to add a mini camera panel, False to add a full camera panel.

        Returns
        -------
        Camera_Panel | None
            The camera panel, or None if the camera could not be added.
        '''
        v_cam = vimba_cam(camera_id)
        v_panel = Vimba_Panel(v_cam, mini, self.get_cam_title(cam))
        CameraList.CAMERAS['Vimba'].append(
            {'Camera': v_cam, 'Panel': v_panel, 'IR': mini}
        )
        return v_panel

    def _add_dummy_camera(self, cam, mini):
        '''
        Add a dummy camera.

        Parameters
        ----------
        cam : dict
            The camera information dictionary.
        mini : bool
            True to add a mini camera panel, False to add a full camera panel.

        Returns
        -------
        Camera_Panel | None
            The camera panel, or None if the camera could not be added.
        '''
        dummy_panel = Dummy_Panel(mini, self.get_cam_title(cam))
        CameraList.CAMERAS['miDummy'].append(
            {'Camera': dummy_panel.cam, 'Panel': dummy_panel, 'IR': mini}
        )
        return dummy_panel

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
        return f'{cam["Model"]} {cam["Serial"]}'

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
                        if isinstance(pan.cam, (IDS_Camera, thorlabs_camera)):
                            pan.cam.free_memory()
                            pan.cam.dispose()
                        if isinstance(item['Camera'], miDummy):
                            miDummy.instances.remove(item['Camera'])

                        pan._dispose_cam = True
                        if pan.acq_job is not None:
                            pan.acq_job.stop_threads = True

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
                if isinstance(panel.cam, (IDS_Camera, thorlabs_camera)):
                    panel.cam.free_memory()
                    panel.cam.dispose()

                # Remove dummy camera instances
                if isinstance(camera_info['Camera'], miDummy):
                    miDummy.instances.remove(camera_info['Camera'])

                # Stop any acquisition jobs
                if panel.acq_job is not None:
                    panel.acq_job.stop_threads = True

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
        self.cam_list = []

        # Add miDummy to the list
        self.cam_list += miDummy.get_camera_list()

        if ueye is not None:
            self.cam_list += IDS_Camera.get_camera_list()

        if os.path.exists(thorlabs_camera.uc480_file):
            self.cam_list += thorlabs_camera.get_camera_list()

        if vb is not None:
            self.cam_list += get_camera_list()

        if self.cam_list is None:
            self.cam_list = []
            print('No cameras connected.')

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
