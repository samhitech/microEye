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
from microEye.hardware.pycromanager.devices import PycroCore, PycroStage
from microEye.hardware.pycromanager.headless import HeadlessInstance, HeadlessManager

#stages
from microEye.hardware.stages.elliptec.devicesView import ElliptecView
from microEye.hardware.stages.elliptec.ellDevices import ELLDevices
from microEye.hardware.stages.kinesis.kinesis import KinesisView, KinesisXY
from microEye.hardware.stages.piezo_concept import PzFoc
from microEye.hardware.stages.stabilizer import FocusStabilizer
from microEye.hardware.stages.stage import ZStageController

#widgets
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
    XY_STAGE = auto()
    Z_STAGE = auto()


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
            self.cameras = []
            self.stages = []
            self.lasers: list[Union[SingleMatchBox, CombinerLaserWidget]] = []

            DeviceManager._initialized = True

    def init_devices(self):
        self._init_cameras_list()
        self._init_ir_cam()
        self._init_laser_relay()
        self._init_z_stage()
        self._init_xy_stage()
        self._init_elliptec_devices()
        self._init_hid_controller()
        self._init_focus_stabilizer()

    def _init_cameras_list(self):
        self.camList = CameraList()
        self.camList.cameraAdded.connect(self._add_camera)
        self.camList.cameraRemoved.connect(self._remove_camera)

        # debounce timer one shot timer
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

    def _init_z_stage(self):
        self.stage: ZStageController = None

        self._set_z_stage('FOC100')

    def _init_xy_stage(self):
        self.stage_xy = KinesisXY()
        view = self.stage_xy.getViewWidget()
        DeviceManager.WIDGETS[DEVICES.XY_STAGE] = view
        WeakObjects.addObject(view)
        self.widgetAdded.emit(DEVICES.XY_STAGE, view)

    def _init_hid_controller(self):
        self.hid_controller = hidController()
        self.hid_controller.reportEvent.connect(self.hid_report)
        # self.hid_controller.reportRStickPosition.connect(self.hid_RStick_report)
        # self.hid_controller.reportLStickPosition.connect(self.hid_LStick_report)
        self.hid_controller_toggle = False

        self.widgetAdded.emit(DEVICES.HID_CONTROLLER, self.hid_controller)

    def _init_focus_stabilizer(self):
        FocusStabilizer.instance().moveStage.connect(self.moveStage)
        FocusStabilizer.instance().startWorker()

        self.focus = focusWidget()
        DeviceManager.WIDGETS[DEVICES.FocusStabilizer] = self.focus
        WeakObjects.addObject(self.focus)

        self.widgetAdded.emit(DEVICES.FocusStabilizer, self.focus)

    def moveStage(self, dir: bool, steps: int):
        if isinstance(self.stage, ZStageController):
            self.stage.moveStage(dir, steps)

    def moveStageXY(self, x: float, y: float):
        if isinstance(self.stage_xy, KinesisXY):
            self.stage_xy.move_relative(x, y)

    def stopRequest(self, axis):
        if axis in ['x', 'y', 'xy']:
            self.stage_xy.stop()

    def homeRequest(self, axis):
        if axis == 'z':
            self.stage.stage.home()

    def toggleLock(self, axis):
        if axis == 'z':
            widget: focusWidget = DeviceManager.WIDGETS.get(DEVICES.FocusStabilizer)
            if widget:
                widget.focusStabilizerView.toggleFocusStabilization()

    def moveRequest(self, axis, direction, step, snap_image=False, relative=True):
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
        relative : bool, optional
            Whether to move relative to the current position (default is True).
        """
        if axis in ['x', 'y']:
            dist = direction * step
            args = [dist, 0] if axis == 'x' else [0, dist]
            if relative:
                self.stage_xy.move_relative(*args)
            else:
                self.stage_xy.move_absolute(*args)
        else:
            if relative:
                self.stage.moveStage(
                    direction > 0, step, interface=True
                ) if self.stage else None
            else:
                self.stage.moveAbsolute(step)

        # debounced snap image
        if snap_image:
            self.timer.start()

    def wait_then_snap(self):
        if self.stage_xy.busy:
            self.timer.start()
        else:
            self.camList.snap_image()

    def hid_report(self, reportedEvent: Buttons):
        self._handle_z_stage_events(reportedEvent)
        self._handle_xy_stage_events(reportedEvent)

    def _handle_z_stage_events(self, reportedEvent: Buttons):
        if isinstance(self.stage, ZStageController):
            if reportedEvent == Buttons.X:
                self.stage.moveStage(True, True, True)
            elif reportedEvent == Buttons.B:
                self.stage.moveStage(False, True, True)
            elif reportedEvent == Buttons.Y:
                self.stage.moveStage(True, False, True)
            elif reportedEvent == Buttons.A:
                self.stage.moveStage(False, False, True)
            elif reportedEvent == Buttons.Options:
                self.stage.stage.home()
            elif reportedEvent == Buttons.R3:
                self.hid_controller_toggle = not self.hid_controller_toggle

    def _handle_xy_stage_events(self, reportedEvent: Buttons):
        kinesisView: KinesisView = DeviceManager.WIDGETS[DEVICES.XY_STAGE]

        if reportedEvent == Buttons.LEFT:
            kinesisView.move(True, self.hid_controller_toggle, False)
        elif reportedEvent == Buttons.RIGHT:
            kinesisView.move(True, self.hid_controller_toggle, True)
        elif reportedEvent == Buttons.UP:
            kinesisView.move(False, self.hid_controller_toggle, True)
        elif reportedEvent == Buttons.DOWN:
            kinesisView.move(False, self.hid_controller_toggle, False)
        elif reportedEvent == Buttons.R1:
            pass
            # kinesisView.center()
        elif reportedEvent == Buttons.L1:
            kinesisView.stop()
        elif reportedEvent == Buttons.L3:
            self._toggle_xy_step_or_jump(kinesisView)

    def _toggle_xy_step_or_jump(self, kinesisView: KinesisView):
        self.hid_controller_toggle = not self.hid_controller_toggle
        kinesisView.updateControls(self.hid_controller_toggle)

    def hid_LStick_report(self, x, y):
        diff_x = x - 128
        diff_y = y - 127
        deadzone = 16
        res = dz_hybrid([diff_x, diff_y], deadzone)
        diff = res[1]

        kinesisView: KinesisView = DeviceManager.WIDGETS[DEVICES.XY_STAGE]

        self._update_step_or_jump_spin(kinesisView, diff)

        if abs(diff) > 0:
            if self.hid_controller_toggle:
                val = 0.0001 * diff
                val += kinesisView.getJump()
                kinesisView.setJump(val)
            else:
                val = 0.0001 * diff
                val += kinesisView.getStep()
                kinesisView.setStep(val)
        else:
            if self.hid_controller_toggle:
                val = kinesisView.getJump()
                val -= val % 0.0005
                kinesisView.setJump(val)
            else:
                val = kinesisView.getJump()
                val -= val % 0.0005
                kinesisView.setJump(val)

    def _update_step_or_jump_spin(self, kinesisView: KinesisView, diff):
        current_value = (
            kinesisView.getJump()
            if self.hid_controller_toggle
            else kinesisView.getStep()
        )
        if abs(diff) > 0:
            new_value = current_value + 0.0001 * diff
        else:
            new_value = current_value - (current_value % 0.0005)
        if self.hid_controller_toggle:
            kinesisView.setJump(new_value)
        else:
            kinesisView.setStep(new_value)

    def hid_RStick_report(self, x, y):
        diff_x = x - 128
        diff_y = y - 127
        deadzone = 16
        res = dz_hybrid([diff_x, diff_y], deadzone)
        diff = res[1]

        if self.hid_controller_toggle:
            self._handle_focus_stabilization(diff)
        else:
            self._handle_z_stage_step(diff)

    def _handle_focus_stabilization(self, diff):
        if FocusStabilizer.instance().isFocusStabilized():
            value = 1e-3 * diff
            FocusStabilizer.instance().setParameter(value, True)

    def _handle_z_stage_step(self, diff):
        if abs(diff) > 0 and abs(diff) < 100:
            value = 0.25 * diff
            self.stage.setStep(value, True)
        else:
            value = self.stage.getStep()
            value -= value % 5
            self.stage.setStep(value)

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

    def _set_z_stage(self, value: str):
        if self.stage and self.stage.isOpen() and self.stage.isSerial():
            QtWidgets.QMessageBox.warning(
                None,
                'Warning',
                f'Please disconnect {self.stage}.',
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        if self.stage:
            self.stage.view.remove_widget()
            self.stage = None

        match = re.search(r'FOC(\d{3})', value)
        if match:
            max_um = int(match.group(1))
            stage = PzFoc(max_um=max_um)
        elif 'Pycromanager' in value and PycroCore._instances.keys().__len__() > 0:
            if len(PycroCore._instances) == 1:
                # If there is only one instance, use it
                stage = PycroStage(port=list(PycroCore._instances.keys())[0])
            else:
                # If there are multiple instances, prompt the user to select one
                port, ok = QtWidgets.QInputDialog.getItem(
                    None,
                    'Select PycroManager Instance',
                    'Select the PycroManager instance to use:',
                    list(map(str, PycroCore._instances.keys())),
                )
                if ok and port:
                    stage = PycroStage(port=int(port))
                else:
                    return
        else:
            return

        self.stage = ZStageController(stage=stage)
        self.stage.view.removed.connect(self._remove_z_stage)
        DeviceManager.WIDGETS[DEVICES.Z_STAGE] = self.stage.view
        WeakObjects.addObject(self.stage.view)
        self.widgetAdded.emit(DEVICES.Z_STAGE, self.stage.view)

    def _remove_z_stage(self, panel: QtWidgets.QWidget):
        # DeviceManager.WIDGETS[DEVICES.Z_STAGE] = None
        WeakObjects.removeObject(panel)
        self.widgetRemoved.emit(DEVICES.Z_STAGE, panel)

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
            'Stage Z': self.stage.get_config(),
            'Stage XY': self.stage_xy.get_config(),
            'FocusStabilizer': self.focus.get_config(),
            'Pycromanager': {
                'core_instances': list(PycroCore._instances.keys()),
                'headless_instances': HeadlessManager()._instances,
            },
        }

        config['Lasers'] = self._get_laser_configs()

        return config

    def load_config(self, config: dict):
        self.laser_relay.load_config(config.get('LaserRelay', {}))
        self.elliptec.load_config(config.get('Elliptec', {}))
        self.stage.load_config(
            config.get('PiezoStage', {})
            if 'PiezoStage' in config
            else config.get('Stage Z', {})
        )
        self.stage_xy.load_config(config.get('Stage XY', {}))
        self.focus.load_config(config.get('FocusStabilizer', {}))

        while len(self.lasers) > 0:
            panel = self.lasers.pop()
            panel.Laser.CloseCOM()
            panel.remove_widget()

        for laser in config.get('Lasers', []):
            laser_widget = self._add_laser(laser.get('class'))
            laser_widget.load_config(laser)

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

    def auto_connect(self):
        funcs = [
            self.laser_relay.connect,
            self.elliptecView.open,
            self.stage.connect,
            self.stage_xy.open,
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
            self.stage.disconnect,
            self.stage_xy.close,
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
