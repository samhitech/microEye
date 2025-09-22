import re
import traceback
import weakref
from enum import Enum, auto
from typing import Union

from microEye.hardware.cams.camera_list import CameraList
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.linescan.IR_Cam import DemoLineScanner, ParallaxLineScanner

# lasers
from microEye.hardware.lasers.io_matchbox import CombinerLaserWidget
from microEye.hardware.lasers.io_single_laser import SingleMatchBox
from microEye.hardware.lasers.laser_relay import LaserRelayController
from microEye.hardware.protocols.actions import WeakObjects

# pycromanager
from microEye.hardware.pycromanager.devices import PycroCore
from microEye.hardware.pycromanager.headless import HeadlessInstance, HeadlessManager
from microEye.hardware.stages.elliptec.devicesView import ElliptecView
from microEye.hardware.stages.elliptec.ellDevices import ELLDevices

# stages
from microEye.hardware.stages.manager import Axis, StageManager, StageManagerView
from microEye.hardware.stages.stabilizer import FocusStabilizer

# widgets
from microEye.hardware.stages.view import StageView
from microEye.hardware.widgets.focusWidget import focusWidget
from microEye.qt import QtCore, QtWidgets, Signal
from microEye.utils.hid_utils.controller import Buttons, dz_hybrid, hidController
from microEye.utils.retry_exec import retry_exec


class DEVICES(Enum):
    CAM_LIST = auto()
    CAMERA = auto()
    ELLIPTEC = auto()
    FocusStabilizer = auto()
    HID_CONTROLLER = auto()
    IR_CAM = auto()
    LASER = auto()
    LASER_RELAY = auto()
    STAGE = auto()
    STAGE_MANAGER = auto()


class DeviceManager(QtCore.QObject):
    widgetAdded = Signal(DEVICES, QtWidgets.QWidget)
    widgetRemoved = Signal(DEVICES, QtWidgets.QWidget)

    WIDGETS = weakref.WeakValueDictionary()
    CONTROLLERS = weakref.WeakValueDictionary()

    _initialized = False

    def __new__(cls, *args, **kwargs):
        # If the single instance doesn't exist, create a new one
        if not hasattr(cls, '_singleton') or cls._singleton is None:
            cls._singleton = super().__new__(cls, *args)
        # Return the single instance
        return cls._singleton

    @classmethod
    def instance(cls):
        if not hasattr(cls, '_singleton') or cls._singleton is None:
            return DeviceManager()

        return cls._singleton

    def __init__(self):
        if not DeviceManager._initialized:
            super().__init__()
            self.lasers: list[Union[SingleMatchBox, CombinerLaserWidget]] = []

            DeviceManager._initialized = True

    def init_devices(self):
        self._init_cameras_list()
        self._init_ir_cam()
        self._init_laser_relay()
        self._init_stages()
        self._init_elliptec_devices()
        self._init_hid_controller()
        self._init_focus_stabilizer()

    def _init_cameras_list(self):
        self.camList = CameraList()
        self.camList.cameraAdded.connect(self._add_camera)
        self.camList.cameraRemoved.connect(self._remove_camera)

        # # debounce timer one shot timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.wait_then_snap)

        self.widgetAdded.emit(DEVICES.CAM_LIST, self.camList)

    def _init_ir_cam(self):
        self.ir_array_detector = None

    def _init_laser_relay(self):
        self.laser_relay = LaserRelayController()
        self.laser_relay.sendCommandActivated.connect(
            lambda: self.laser_relay.sendCommand(self.laser_relay_settings())
        )
        DeviceManager.WIDGETS[DEVICES.LASER_RELAY] = self.laser_relay.view
        WeakObjects.addObject(self.laser_relay)
        self.widgetAdded.emit(DEVICES.LASER_RELAY, self.laser_relay.view)

    def _init_elliptec_devices(self):
        self._add_elliptec_devices()

    def _add_elliptec_devices(self):
        self.elliptec = ELLDevices()
        self.elliptecView = ElliptecView(elleptic=self.elliptec)
        DeviceManager.WIDGETS[DEVICES.ELLIPTEC] = self.elliptecView
        WeakObjects.addObject(self.elliptecView)
        self.widgetAdded.emit(DEVICES.ELLIPTEC, self.elliptecView)

    def _init_stages(self):
        instance = StageManager()
        instance.stageAdded.connect(self._set_stage)

        self.stageManagerView = StageManagerView()
        DeviceManager.WIDGETS[DEVICES.STAGE_MANAGER] = self.stageManagerView
        WeakObjects.addObject(self.stageManagerView)
        self.widgetAdded.emit(DEVICES.STAGE_MANAGER, self.stageManagerView)

    def _init_hid_controller(self):
        self.hid_controller = hidController()
        self.hid_controller.reportEvent.connect(self.hid_report)
        # self.hid_controller.reportRStickPosition.connect()
        # self.hid_controller.reportLStickPosition.connect()
        self.hid_controller_toggle = False

        self.widgetAdded.emit(DEVICES.HID_CONTROLLER, self.hid_controller)

    def _init_focus_stabilizer(self):
        FocusStabilizer.instance().moveStage.connect(StageManager.instance().move_z)
        FocusStabilizer.instance().startWorker()

        self.focus = focusWidget()
        DeviceManager.WIDGETS[DEVICES.FocusStabilizer] = self.focus
        WeakObjects.addObject(self.focus)

        self.widgetAdded.emit(DEVICES.FocusStabilizer, self.focus)

    def stopRequest(self, axis: Axis):
        StageManager.instance().stop(axis)

    def homeRequest(self, axis: Axis):
        StageManager.instance().home(axis)

    def toggleLock(self, axis: Axis):
        if axis == Axis.Z:
            widget: focusWidget = DeviceManager.WIDGETS.get(DEVICES.FocusStabilizer)
            if widget:
                widget.focusStabilizerView.toggleFocusStabilization()

    def moveRequest(self, axis: Axis, direction: bool, jump: bool, snap_image=False):
        """
        Move the stage in the specified direction and step size.

        Parameters
        ----------
        axis : str
            The axis to move ('x', 'y', or 'z').
        direction : bool
            The direction to move (True for positive, False for negative).
        step : int
            The step size for the movement.
        """
        if axis in [Axis.X, Axis.Y]:
            StageManager.instance().move_xy(axis == Axis.X, jump, direction)
        elif axis == Axis.Z:
            StageManager.instance().move_z(direction, jump, True)

        # debounced snap image
        if snap_image:
            self.timer.start()

    def wait_then_snap(self):
        if StageManager.instance().is_busy(Axis.X):
            self.timer.start()
        else:
            self.camList.snap_image()

    def hid_report(self, reportedEvent: Buttons):
        self._handle_z_stage_events(reportedEvent)
        self._handle_xy_stage_events(reportedEvent)

    def _handle_z_stage_events(self, reportedEvent: Buttons):
        mapping = {
            Buttons.X: (True, True),
            Buttons.B: (False, True),
            Buttons.Y: (True, False),
            Buttons.A: (False, False),
        }
        if reportedEvent in mapping:
            StageManager.instance().move_z(*mapping[reportedEvent], interface=True)
        elif reportedEvent == Buttons.Options:
            StageManager.instance().home(Axis.Z)
        elif reportedEvent == Buttons.R3:
            self.hid_controller_toggle = not self.hid_controller_toggle

    def _handle_xy_stage_events(self, reportedEvent: Buttons):
        instance = StageManager.instance()

        mapping = {
            Buttons.LEFT: (True, self.hid_controller_toggle, False),
            Buttons.RIGHT: (True, self.hid_controller_toggle, True),
            Buttons.UP: (False, self.hid_controller_toggle, True),
            Buttons.DOWN: (False, self.hid_controller_toggle, False),
        }

        if reportedEvent in mapping:
            instance.move_xy(*mapping[reportedEvent])
        elif reportedEvent == Buttons.R1:
            pass
        elif reportedEvent == Buttons.L1:
            instance.stop(Axis.X)
        elif reportedEvent == Buttons.L3:
            self._toggle_xy_step_or_jump()

    def _toggle_xy_step_or_jump(self):
        self.hid_controller_toggle = not self.hid_controller_toggle
        # kinesisView.updateControls(self.hid_controller_toggle)

    # gui + hardware

    def _add_laser(self, laser: str):
        if (
            'IO MatchBox' in laser and 'Combiner' in laser
        ) or CombinerLaserWidget.__name__ in laser:
            widget = CombinerLaserWidget()
        elif (
            'IO MatchBox' in laser and 'Single' in laser
        ) or SingleMatchBox.__name__ in laser:
            widget = SingleMatchBox()
        else:
            raise ValueError(f'Unknown laser type: {laser}')

        self.lasers.append(widget)
        WeakObjects.addObject(widget)

        widget.removed.connect(lambda: self._laser_removed(widget))

        self.widgetAdded.emit(DEVICES.LASER, widget)

        return widget

    def _laser_removed(self, widget: QtWidgets.QWidget):
        WeakObjects.removeObject(widget)
        if widget in self.lasers:
            self.lasers.remove(widget)
        self.widgetRemoved.emit(DEVICES.LASER, widget)

    def _add_camera(self, widget: Camera_Panel, ir: bool):
        if ir:
            widget._frames = FocusStabilizer.instance().buffer

            self.widgetAdded.emit(DEVICES.IR_CAM, widget)

        else:
            widget.setWindowTitle(widget.title())
            widget.show()

        WeakObjects.addObject(widget)

    def _remove_camera(self, widget: QtWidgets.QWidget, ir: bool):
        WeakObjects.removeObject(widget)
        if ir:
            self.widgetRemoved.emit(DEVICES.IR_CAM, widget)

    def _set_ir_array_detector(self, value: str):
        if self.camList.autofocusCam:
            title = self.camList.autofocusCam.title()
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                f'Please remove {title}.',
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        if self.ir_array_detector and self.ir_array_detector.isOpen:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                f'Please disconnect {self.ir_array_detector.name}.',
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        if 'TSL1401' in value:
            self.ir_array_detector = ParallaxLineScanner(
                FocusStabilizer.instance().buffer
            )
            widget = self.ir_array_detector.getQWidget()
            self.widgetAdded.emit(DEVICES.IR_CAM, widget)
            DeviceManager.WIDGETS[DEVICES.IR_CAM] = widget
        elif 'Demo Line Scanner' in value:
            self.ir_array_detector = DemoLineScanner(FocusStabilizer.instance().buffer)
            widget = self.ir_array_detector.getQWidget()
            self.widgetAdded.emit(DEVICES.IR_CAM, widget)
            DeviceManager.WIDGETS[DEVICES.IR_CAM] = widget

    def _remove_ir_array_detector(self):
        if self.ir_array_detector.isOpen:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                f'Please disconnect {self.ir_array_detector.name}.',
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        self.widgetRemoved.emit(DEVICES.IR_CAM, DeviceManager.WIDGETS[DEVICES.IR_CAM])
        self.ir_array_detector = None

    def _set_stage(self, key: str):
        stage = StageManager.STAGES.get(key)

        if stage is None:
            return

        view = StageView(stage=stage)

        DeviceManager.WIDGETS[key] = view
        WeakObjects.addObject(view)
        self.widgetAdded.emit(DEVICES.STAGE, view)

    def laser_relay_settings(self):
        '''Returns the RelayBox setting command.

        Returns
        -------
        str
            the RelayBox setting command.
        '''
        config = ''
        for panel in self.lasers:
            config += panel.GetRelayState()
        return self.laser_relay.getCommand(config)

    def _get_laser_configs(self) -> dict:
        '''
        Returns the laser configurations.

        Returns
        -------
        list
            list of laser configurations.
        '''
        configs = []
        for panel in self.lasers:
            configs.append(panel.get_config())

        return configs

    def get_config(self) -> dict:
        config = {
            'LaserRelay': self.laser_relay.get_config(),
            'Elliptec': self.elliptec.get_config(),
            'Stages': StageManager.instance().get_config(),
            'FocusStabilizer': self.focus.get_config(),
            'Pycromanager': {
                'core_instances': list(PycroCore._instances.keys()),
                'headless_instances': HeadlessManager()._instances,
            },
            'Cameras': self.camList.get_config(),
        }

        config['Lasers'] = self._get_laser_configs()

        return config

    def load_config(self, config: dict):
        try:
            pycro_config: dict = config.get('Pycromanager', {})

            core_instances: list[int] = pycro_config.get('core_instances', [])
            for port in core_instances:
                PycroCore.instance(int(port))

            headless_instances: dict[int, dict] = pycro_config.get(
                'headless_instances', {}
            )
            for _, instance in headless_instances.items():
                HeadlessManager().start_instance(HeadlessInstance(**instance))
        except Exception:
            traceback.print_exc()

        self.laser_relay.load_config(config.get('LaserRelay', {}))
        self.elliptec.load_config(config.get('Elliptec', {}))
        StageManager.instance().load_config(config.get('Stages', {}))
        self.focus.load_config(config.get('FocusStabilizer', {}))

        while len(self.lasers) > 0:
            panel = self.lasers.pop()
            panel.Laser.CloseCOM()
            panel.remove_widget()

        for laser in config.get('Lasers', []):
            laser_widget = self._add_laser(laser.get('class'))
            laser_widget.load_config(laser)

        self.camList.load_config(config.get('Cameras', []))

    def auto_connect(self):
        funcs = [
            self.laser_relay.connect,
            self.elliptecView.open,
            StageManager.instance().open_all,
        ]

        for panel in self.lasers:
            funcs.append(panel.laser_connect)

        for func in funcs:
            retry_exec(func)

    def shutdown(self, exit=True):
        '''Disconnects all devices and exits the application.'''

        if self.camList.autofocusCam:
            self.camList.autofocusCam.stop()

        funcs = [
            self.camList.removeAllCameras,
            self.laser_relay.disconnect,
            self.elliptecView.close,
            StageManager.instance().close_all,
        ]

        for panel in self.lasers:
            funcs.append(panel.Laser.CloseCOM)

        for func in funcs:
            retry_exec(func)

        print('All devices disconnected!')
        if not exit:
            return

        import sys

        sys.exit(0)
