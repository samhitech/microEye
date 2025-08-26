import os
import threading
import traceback

import cv2
import numpy as np
import pyqtgraph as pg
import tabulate
import tifffile as tf

# Cameras
from microEye.hardware.cams.camera_list import CameraList
from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.micam import miCamera
from microEye.hardware.cams.vimba import vimba_cam

# devices manager
from microEye.hardware.mieye.devices_manager import DeviceManager

# other imports
from microEye.hardware.protocols.actions import WeakObjects
from microEye.hardware.stages.stabilizer import FocusStabilizer
from microEye.hardware.widgets.scan_acquisition import (
    ScanAcquisitionWidget,
    TiledImageSelector,
    TileImage,
)
from microEye.qt import (
    QtCore,
    Signal,
)
from microEye.utils.thread_worker import QThreadWorker
from microEye.utils.uImage import uImage


class AcquisitionManager(QtCore.QObject):
    acquisitionStarted = Signal()
    acquisitionFinished = Signal(object)

    def __init__(self):
        super().__init__()

        self.acquisitionWidget = ScanAcquisitionWidget()
        WeakObjects.addObject(self.acquisitionWidget)

        self._init_acquisition_signals()

        self.lastTile = None
        self.scan_worker = None
        self._stop_scan = False
        self._scanning = False

    def _init_acquisition_signals(self):
        self.acquisitionWidget.startAcquisitionXY.connect(self.start_scan_acquisitionXY)
        self.acquisitionWidget.startAcquisitionZ.connect(self.start_scan_acquisitionZ)
        self.acquisitionWidget.startCalibrationZ.connect(self.start_calibration_Z)
        self.acquisitionWidget.stopAcquisitionXY.connect(self.stop_scan_acquisition)
        self.acquisitionWidget.stopAcquisitionZ.connect(self.stop_scan_acquisition)
        self.acquisitionWidget.openLastTileXY.connect(self.show_last_tile)
        self.acquisitionWidget.directoryChanged.connect(self.update_directories)
        self.acquisitionWidget.moveZ.connect(DeviceManager.instance().moveStage)

    def result_scan_acquisition(self, data):
        self._scanning = False
        self.acquisitionWidget.setActionsStatus(True)

        if data:
            self.lastTile = TiledImageSelector(data)
            self._connect_tile_position_signal(self.lastTile)
            self.lastTile.show()

    def _connect_tile_position_signal(self, tile: TiledImageSelector):
        tile.positionSelected.connect(
            lambda x, y: DeviceManager.instance().stage_xy.move_absolute(x, y)
        )

    def result_z_calibration(self, data):
        self._scanning = False
        self.acquisitionWidget.setActionsStatus(True)

        if data is not None:
            coeff = np.polyfit(data[:, 0], data[:, 1], 1)
            FocusStabilizer.instance().setCalCoeff(coeff[0])
            plot_z_cal(data, coeff)

    def result_scan_export(self, data: list[TileImage]):
        self._scanning = False
        self.acquisitionWidget.setActionsStatus(True)

        if data:
            self._export_tile_images(data)

    def _export_tile_images(self, data: list[TileImage]):
        directory = self.acquisitionWidget._directory
        if len(directory) > 0:
            index = self._get_next_index(directory)
            path = os.path.join(directory, f'{index:03d}_XY/')
            os.makedirs(path, exist_ok=True)

            for idx, tile_img in enumerate(data):
                filename = (
                    f'{idx:03d}_image_y{tile_img.index[0]:02d}'
                    + f'_x{tile_img.index[1]:02d}.tif'
                )
                tf.imwrite(
                    os.path.join(path, filename),
                    tile_img.uImage.image,
                    photometric='minisblack',
                )

    def _get_next_index(self, directory):
        index = 0
        while os.path.exists(os.path.join(directory, f'{index:03d}_XY/')):
            index += 1
        return index

    def start_scan_acquisitionXY(self, params):
        if not self._scanning:
            self._stop_scan = False
            self._scanning = True

            self.scan_worker = QThreadWorker(
                scanAcquisition,
                [params[0], params[1]],
                [params[2], params[3]],
                params[4],
                params[5],
            )
            self.scan_worker.signals.result.connect(self.result_scan_acquisition)
            # Execute
            QtCore.QThreadPool.globalInstance().start(self.scan_worker)

            self.acquisitionWidget.setActionsStatus(False)

    def start_scan_acquisitionZ(self, params):
        if not self._scanning:
            self._stop_scan = False
            self._scanning = True

            self.scan_worker = QThreadWorker(
                z_stack_acquisition,
                self,
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
            )
            self.scan_worker.signals.result.connect(self.result_scan_acquisition)
            # Execute
            QtCore.QThreadPool.globalInstance().start(self.scan_worker)

            self.acquisitionWidget.setActionsStatus(False)

    def start_calibration_Z(self, params):
        if not self._scanning:
            self._stop_scan = False
            self._scanning = True

            self.scan_worker = QThreadWorker(
                z_calibration,
                self,
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
            )
            self.scan_worker.signals.result.connect(self.result_z_calibration)
            # Execute
            QtCore.QThreadPool.globalInstance().start(self.scan_worker)

            self.acquisitionWidget.setActionsStatus(False)

    def stop_scan_acquisition(self):
        self._stop_scan = True
        if self.scan_worker:
            self.scan_worker.stop()

    def update_directories(self, value: str):
        for _, cam_list in CameraList.CAMERAS.items():
            for cam in cam_list:
                panel: Camera_Panel = cam['Panel']
                panel._directory = value
                panel.camera_options.set_param_value(CamParams.SAVE_DIRECTORY, value)

    def show_last_tile(self):
        if self.lastTile is not None:
            self.lastTile.show()


def scanAcquisition(steps, step_size, delay, average=1, **kwargs):
    '''Scan Acquisition (works with Allied Vision Cams only)

    Parameters
    ----------
    steps : (int, int)
        number of grid steps (x ,y)
    step_size : (float, float)
        step size in um (x ,y)
    delay : float
        delay in ms after moving before acquiring images
    average : int
        number of frames to average, default 1 (no averaging)
    **kwargs : dict
        additional arguments
        - event : `threading.Event`
            event to stop the acquisition

    Returns
    -------
    list[TileImage]
        result data list of TileImages
    '''
    try:
        event: threading.Event = kwargs.get('event')
        if event and event.is_set():
            return

        device_manager = DeviceManager.instance()

        data = []
        vimba_cams = [cam for cam in CameraList.CAMERAS['Vimba'] if not cam['IR']]
        if device_manager.stage_xy.isOpen() and len(vimba_cams) > 0:
            cam: vimba_cam = vimba_cams[0]['Camera']
            for x in range(steps[0]):
                device_manager.stage_xy.move_relative(
                    round(step_size[0] / 1000, 4), 0, False
                )
                for y in range(steps[1]):
                    if event and event.is_set():
                        return

                    if y > 0:
                        device_manager.stage_xy.move_relative(
                            0, ((-1) ** x) * round(step_size[1] / 1000, 4), False
                        )
                    frame = None
                    with cam.cam:
                        QtCore.QThread.msleep(delay)
                        if average > 1:
                            frames_avg = []
                            for _n in range(average):
                                frames_avg.append(
                                    cam.cam.get_frame().as_numpy_ndarray()[..., 0]
                                )
                            frame = uImage(
                                np.array(frames_avg).mean(axis=0, dtype=np.uint16)
                            )
                        else:
                            frame = uImage(
                                cam.cam.get_frame().as_numpy_ndarray()[..., 0]
                            )
                        frame.equalizeLUT(None, True)
                    frame._view = cv2.resize(
                        frame._view,
                        (0, 0),
                        fx=0.5,
                        fy=0.5,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    Y = (x % 2) * (steps[1] - 1) + ((-1) ** x) * y
                    data.append(
                        TileImage(frame, [Y, x], device_manager.stage_xy.position)
                    )
                    cv2.imshow(cam.name, frame._view)
                    cv2.waitKey(1)

            # device_manager.stage_xy.update()
        else:
            return
    except Exception:
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            traceback.print_exc()

    return data


def z_stack_acquisition(
    acquisition_manager: AcquisitionManager,
    n,
    step_size,
    delay=100,
    nFrames=1,
    reverse=False,
    **kwargs,
):
    '''Z-Stack Acquisition (works with Allied Vision Cams only)

    Parameters
    ----------
    acq_manager : AcquisitionManager
        acquisition manager instance
    n : int
        number of z-stacks
    step_size : int
        step size in nm along z-axis
    delay : float
        delay in ms after moving before acquiring images
    nFrames : int
        number of frames for each stack
    reverse : bool
        reverse the direction of the movement
    **kwargs : dict
        additional arguments
        - event : `threading.Event`
            event to stop the acquisition
    '''
    try:
        event: threading.Event = kwargs.get('event')
        if event and event.is_set():
            return

        device_manager = DeviceManager.instance()

        data = []
        peak = None
        vimba_cams = [cam for cam in CameraList.CAMERAS['Vimba'] if not cam['IR']]
        if device_manager.stage.isOpen() and len(vimba_cams) > 0 and nFrames > 0:
            cam: miCamera = vimba_cams[0]['Camera']
            cam_pan: Camera_Panel = vimba_cams[0]['Panel']
            if cam.acquisition:
                return

            cam_pan.camera_options.set_param_value(CamParams.FRAMES, nFrames)
            cam_pan.camera_options.set_param_value(CamParams.SAVE_DATA, True)

            peak = FocusStabilizer.instance().getParameter()
            for x in range(n):
                if x > 0:
                    if event and event.is_set():
                        return

                    if (
                        FocusStabilizer.instance().isFocusStabilized()
                        and FocusStabilizer.instance().useCal()
                    ):
                        value = FocusStabilizer.instance().calCoeff() * step_size
                        if reverse:
                            value *= -1
                        FocusStabilizer.instance().setParameter(value, True)
                        QtCore.QThread.msleep(delay)
                    else:
                        if FocusStabilizer.instance().isFocusStabilized():
                            FocusStabilizer.instance().toggleFocusStabilization(False)
                        acquisition_manager.acquisitionWidget.moveZ.emit(
                            reverse, step_size
                        )
                        QtCore.QThread.msleep(delay)
                        FocusStabilizer.instance().toggleFocusStabilization(True)
                frame = None
                prefix = f'Z_{x:04d}_'

                cam_event = threading.Event()

                cam_pan.asyncFreerun.emit(prefix, cam_event)

                cam_event.wait()
                QtCore.QThread.msleep(100)
        else:
            print('Z-scan failed!')
            info = [
                {
                    'Z-Stage Open': device_manager.stage.isOpen(),
                    'Camera Available': len(vimba_cams) > 0,
                    'Frames > 0': nFrames > 0,
                }
            ]
            print(tabulate.tabulate(info, headers='keys', tablefmt='rounded_grid'))
    except Exception:
        traceback.print_exc()
    finally:
        if peak:
            FocusStabilizer.instance().setParameter(peak)
    return


def z_calibration(
    acquisition_manager: AcquisitionManager,
    n,
    step_size,
    delay=100,
    nFrames=50,
    reverse=False,
    **kwargs,
):
    '''Z-Stack Acquisition (works with Allied Vision Cams only)

    Parameters
    ----------
    acq_manager : AcquisitionManager
        acquisition manager instance
    n : int
        number of z-stacks
    step_size : int
        step size in nm along z-axis
    delay : float
        delay in ms per measurement
    nFrames : int
        number of frames used for each measurement
    reverse : bool
        reverse the direction of the movement
    **kwargs : dict
        additional arguments
        - event : `threading.Event`
            event to stop the acquisition
    '''
    positions = np.zeros((n, 2))
    try:
        event: threading.Event = kwargs.get('event')
        if event and event.is_set():
            return

        device_manager = DeviceManager.instance()

        data = []
        if device_manager.stage.isOpen():
            if FocusStabilizer.instance().isFocusStabilized():
                FocusStabilizer.instance().toggleFocusStabilization(False)
            for x in range(n):
                if event and event.is_set():
                    return

                if x > 0:
                    acquisition_manager.acquisitionWidget.moveZ.emit(reverse, step_size)
                QtCore.QThread.msleep(delay * nFrames)
                positions[x, 0] = x * step_size
                positions[x, 1] = np.mean(
                    FocusStabilizer.instance().parameter_buffer[-nFrames:]
                )
    except Exception:
        traceback.print_exc()
        positions = None
    return positions


def plot_z_cal(data, coeff):
    model = np.poly1d(coeff)

    x = np.linspace(data[0, 0], data[-1, 0], 1001)

    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties of the label for y axis
    plt.setLabel('left', 'Central Pixel', units='')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'Z [nm]', units='')

    plt.setWindowTitle(f'Slope: {coeff[0]} | Intercept {coeff[1]}')

    # setting horizontal range
    plt.setXRange(data[0, 0], data[-1, 0])

    # setting vertical range
    plt.setYRange(data[0, 1], data[-1, 1])

    line1 = plt.plot(
        data[:, 0],
        data[:, 1],
        pen='g',
        symbol='x',
        symbolPen='g',
        symbolBrush=0.2,
        name='Data',
    )
    line2 = plt.plot(x, model(x), pen='b', name='Fit')
