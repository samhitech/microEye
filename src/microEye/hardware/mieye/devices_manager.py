import weakref
from enum import Enum

from microEye.hardware.cams import *
from microEye.hardware.lasers import *
from microEye.hardware.protocols import WeakObjects
from microEye.hardware.stages import *
from microEye.qt import QtCore
from microEye.utils.hid import Buttons, dz_hybrid, hidController


class DeviceManager(QtCore.QObject):
    class DEVICES(Enum):
        ELLIPTEC = 0
        LASER_RELAY = 1
        Z_STAGE = 2
        XY_STAGE = 3

    WIDGETS = weakref.WeakValueDictionary()

    def __init__(self, mieye):
        super().__init__()
        self.mieye = mieye
        self.cameras = []
        self.stages = []
        self.lasers = []
        self.controllers = []

        self.init_devices()

    def init_devices(self):
        self._init_ir_cam()
        self._init_laser_relay()
        self._init_elliptec_devices()
        self._init_z_stage()
        self._init_xy_stage()
        self._init_hid_controller()
        self._init_focus_stabilizer()

    def _init_ir_cam(self):
        self.IR_Cam = IR_Cam()

    def _init_laser_relay(self):
        self.laserRelayCtrllr = LaserRelayController()
        self.laserRelayCtrllr.sendCommandActivated.connect(
            lambda: self.laserRelayCtrllr.sendCommand(self.mieye.getRelaySettings())
        )
        DeviceManager.WIDGETS[DeviceManager.DEVICES.LASER_RELAY] = (
            self.laserRelayCtrllr.view
        )
        WeakObjects.addObject(self.laserRelayCtrllr)

    def _init_elliptec_devices(self):
        self.elliptecView = ElliptecView()
        DeviceManager.WIDGETS[DeviceManager.DEVICES.ELLIPTEC] = self.elliptecView
        WeakObjects.addObject(self.elliptecView)

    def _init_z_stage(self):
        self.stage: PzFocController = None

    def _init_xy_stage(self):
        self.kinesisXY = KinesisXY()
        self.kinesisXY_view = self.kinesisXY.getViewWidget()
        DeviceManager.WIDGETS[DeviceManager.DEVICES.XY_STAGE] = (
            self.kinesisXY_view
        )
        WeakObjects.addObject(self.kinesisXY_view)

    def _init_hid_controller(self):
        self.hid_controller = hidController()
        self.hid_controller.reportEvent.connect(self.hid_report)
        # self.hid_controller.reportRStickPosition.connect(self.hid_RStick_report)
        # self.hid_controller.reportLStickPosition.connect(self.hid_LStick_report)
        self.hid_controller_toggle = False

    def _init_focus_stabilizer(self):
        FocusStabilizer.instance().moveStage.connect(self.moveStage)
        FocusStabilizer.instance().startWorker()

    def moveStage(self, dir: bool, steps: int):
        if isinstance(self.stage, PzFocController):
            self.stage.moveStage(dir, steps)

    def moveRequest(self, axis, direction, step):
        if axis in ['x', 'y']:
            dist = direction * step
            args = [dist, 0] if axis == 'x' else [0, dist]
            self.kinesisXY.move_relative(*args)
        else:
            self.stage.moveStage(direction, step) if self.stage else None

    def hid_report(self, reportedEvent: Buttons):
        self._handle_z_stage_events(reportedEvent)
        self._handle_xy_stage_events(reportedEvent)

    def _handle_z_stage_events(self, reportedEvent: Buttons):
        if isinstance(self.stage, PzFocController):
            if reportedEvent == Buttons.X:
                self.stage.moveStage(True, True, True)
            elif reportedEvent == Buttons.B:
                self.stage.moveStage(False, True, True)
            elif reportedEvent == Buttons.Y:
                self.stage.moveStage(True, False, True)
            elif reportedEvent == Buttons.A:
                self.stage.moveStage(False, False, True)
            elif reportedEvent == Buttons.Options:
                self.stage.stage.HOME()
            elif reportedEvent == Buttons.R3:
                self.hid_controller_toggle = not self.hid_controller_toggle

    def _handle_xy_stage_events(self, reportedEvent: Buttons):
        kinesisView: KinesisView = DeviceManager.WIDGETS[DeviceManager.DEVICES.XY_STAGE]

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

        kinesisView: KinesisView = DeviceManager.WIDGETS[DeviceManager.DEVICES.XY_STAGE]

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
            FocusStabilizer.instance().setPeakPosition(value, True)

    def _handle_z_stage_step(self, diff):
        if abs(diff) > 0 and abs(diff) < 100:
            value = 0.25 * diff
            self.stage.setStep(value, True)
        else:
            value = self.stage.getStep()
            value -= value % 5
            self.stage.setStep(value)
