import logging
import time
import traceback
from enum import Enum

from microEye.analysis.tools.roi_selectors import (
    MultiRectangularROISelector,
    convert_pos_size_to_rois,
    convert_rois_to_pos_size,
)
from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.micam import miDummy
from microEye.qt import QDateTime, QtCore, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.gui_helper import get_scaling_factor
from microEye.utils.metadata_tree import MetaParams
from microEye.utils.thread_worker import QThreadWorker
from microEye.utils.uImage import uImage

try:
    import vimba as vb
except Exception:
    vb = None


class DummyParams(Enum):
    FREERUN = 'Acquisition.Freerun'
    STOP = 'Acquisition.Stop'
    LOAD = 'Acquisition Settings.Load Config'
    SAVE = 'Acquisition Settings.Save Config'
    GAIN = 'Acquisition Settings.Gain'
    OFFSET = 'Acquisition Settings.Offset'
    HEIGHT = 'Acquisition Settings.Height'
    WIDTH = 'Acquisition Settings.Width'
    BINNING_HORIZONTAL = 'Acquisition Settings.Binning Horizontal'
    BINNING_VERTICAL = 'Acquisition Settings.Binning Vertical'
    BIT_DEPTH = 'Acquisition Settings.Bit Depth'
    FULL_WELL_CAPACITY = 'Acquisition Settings.Full Well Capacity'
    QUANTUM_EFFICIENCY = 'Acquisition Settings.Quantum Efficiency'
    DARK_CURRENT = 'Acquisition Settings.Dark Current'
    READOUT_NOISE = 'Acquisition Settings.Readout Noise'
    NOISE_BASELINE = 'Acquisition Settings.Noise Baseline'
    FLUX = 'Acquisition Settings.Flux'
    PATTERN_TYPE = 'Acquisition Settings.Pattern Type'
    PATTERN_OFFSET = 'Acquisition Settings.Pattern Offset'
    PATTERN_SINUSOIDAL = 'Acquisition Settings.Sinusoidal Pattern'
    SINUSOIDAL_FREQUENCY = 'Acquisition Settings.Sinusoidal Pattern.Frequency'
    SINUSOIDAL_PHASE = 'Acquisition Settings.Sinusoidal Pattern.Phase'
    SINUSOIDAL_AMPLITUDE = 'Acquisition Settings.Sinusoidal Pattern.Amplitude'
    SINUSOIDAL_DIRECTION = 'Acquisition Settings.Sinusoidal Pattern.Direction'

    DRIFT_SIM = 'Acquisition Settings.Drift Simulation'
    DRIFT_SPEED = DRIFT_SIM + '.Speed [pixel/frames]'
    DRIFT_PERIOD = DRIFT_SIM + '.Period [frames]'
    DRIFT_MOMENTUM = DRIFT_SIM + '.Momentum Coefficient'
    DRIFT_RANDOM_WALK = DRIFT_SIM + '.Random Walk Coefficient'
    DRIFT_VIBRATION = DRIFT_SIM + '.Vibration Amplitude'
    DRIFT_SIGMA_X = DRIFT_SIM + r'.$\sigma_x$ [pixels]'
    DRIFT_SIGMA_Y = DRIFT_SIM + r'.$\sigma_y$ [pixels]'
    DRIFT_AMPLITUDE = DRIFT_SIM + '.Amplitude'

    SM = 'Acquisition Settings.Single Molecule Sim'
    SM_INTENSITY = SM + '.Intensity [photons/loc]'
    SM_DENSITY = SM + '.Density [loc/um]'
    SM_WAVELENGTH = SM + '.Wavelength [nm]'
    SM_PIXEL_SIZE = SM + '.Projected Pixel Size [nm]'
    SM_NA = SM + '.Objective NA'

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


class Dummy_Panel(Camera_Panel):
    '''
    A Qt Widget for a dummy camera
     | Inherits Camera_Panel
    '''

    PARAMS = DummyParams

    def __init__(self, mini=False, *args, **kwargs):
        '''
        Initializes a new Dummy_Panel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        mini : bool, optional
            Flag indicating if this is a mini camera panel, by default False.

        Other Parameters
        ---------------
        *args
            Arguments to pass to the Camera_Panel constructor.

        **kwargs
            Keyword arguments to pass to the Camera_Panel constructor.
        '''
        super().__init__(miDummy(), mini, *args, **kwargs)

        self.OME_tab.set_param_value(MetaParams.CHANNEL_NAME, self._cam.name)
        self.OME_tab.set_param_value(MetaParams.DET_MANUFACTURER, 'MicroEye')
        self.OME_tab.set_param_value(MetaParams.DET_MODEL, 'Dummy')
        self.OME_tab.set_param_value(MetaParams.DET_SERIAL, '123456789')
        self.OME_tab.set_param_value(MetaParams.DET_TYPE, 'CMOS')

        # exposure init
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits((self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(
            step=self._cam.exposure_increment, suffix=self._cam.exposure_unit
        )
        exposure.setValue(self._cam.exposure_current)
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        HEIGHT = {
            'name': str(DummyParams.HEIGHT),
            'type': 'int',
            'value': 512,
            'limits': [0, 4096],
        }
        WIDTH = {
            'name': str(DummyParams.WIDTH),
            'type': 'int',
            'value': 512,
            'limits': [0, 4096],
        }
        BINNING_HORIZONTAL = {
            'name': str(DummyParams.BINNING_HORIZONTAL),
            'type': 'int',
            'value': 1,
            'limits': [0, 10],
        }
        BINNING_VERTICAL = {
            'name': str(DummyParams.BINNING_VERTICAL),
            'type': 'int',
            'value': 1,
            'limits': [0, 10],
        }
        BIT_DEPTH = {
            'name': str(DummyParams.BIT_DEPTH),
            'type': 'list',
            'limits': [8, 10, 12, 16],
            'value': 12,
        }
        GAIN = {'name': str(DummyParams.GAIN), 'type': 'float', 'value': 2.23}
        FULL_WELL_CAPACITY = {
            'name': str(DummyParams.FULL_WELL_CAPACITY),
            'type': 'int',
            'value': 9200,
            'suffix': 'e-',
        }
        QUANTUM_EFFICIENCY = {
            'name': str(DummyParams.QUANTUM_EFFICIENCY),
            'type': 'float',
            'value': 0.8,
            'limits': [0, 1],
        }
        DARK_CURRENT = {
            'name': str(DummyParams.DARK_CURRENT),
            'type': 'float',
            'value': 0.0001,
            'suffix': ' e-/s',
        }
        READOUT_NOISE = {
            'name': str(DummyParams.READOUT_NOISE),
            'type': 'float',
            'value': 2.1,
            'suffix': ' e-',
        }
        NOISE_BASELINE = {
            'name': str(DummyParams.NOISE_BASELINE),
            'type': 'float',
            'value': 5.0,
            'limits': [0, 2**16],
            'suffix': ' ADU',
        }
        FLUX = {
            'name': str(DummyParams.FLUX),
            'type': 'float',
            'value': 0,
            'suffix': ' e-/p/s',
        }
        PATTERN_TYPE = {
            'name': str(DummyParams.PATTERN_TYPE),
            'type': 'list',
            'limits': [
                'Constant Flux',
                'Sinusoidal',
                'Gaussian',
                'Astigmatic Fiducials',
                'Single Molecule Sim',
            ],
            'value': 'Sinusoidal',
        }
        PATTERN_OFFSET = {
            'name': str(DummyParams.PATTERN_OFFSET),
            'type': 'float',
            'value': 0.0,
            'suffix': ' e-',
        }
        PATTERN_SINUSOIDAL = {
            'name': str(DummyParams.PATTERN_SINUSOIDAL),
            'type': 'group',
            'expanded': False,
            'children': [
                {
                    'name': str(DummyParams.SINUSOIDAL_FREQUENCY),
                    'type': 'float',
                    'value': 0.01,
                    'suffix': ' Hz',
                },
                {
                    'name': str(DummyParams.SINUSOIDAL_PHASE),
                    'type': 'float',
                    'value': 0.0,
                    'suffix': ' deg',
                },
                {
                    'name': str(DummyParams.SINUSOIDAL_AMPLITUDE),
                    'type': 'float',
                    'value': 0.1e5,
                    'suffix': ' e-/p/s',
                },
                {
                    'name': str(DummyParams.SINUSOIDAL_DIRECTION),
                    'type': 'list',
                    'limits': ['d', 'h', 'v', 'r'],
                    'value': 'd',
                },
            ],
        }
        SM_SIM = {
            'name': str(DummyParams.SM),
            'type': 'group',
            'expanded': False,
            'children': [
                {
                    'name': str(DummyParams.SM_INTENSITY),
                    'type': 'float',
                    'value': 5000,
                    'decimals': 6,
                },
                {
                    'name': str(DummyParams.SM_DENSITY),
                    'type': 'float',
                    'value': 0.5,
                    'decimals': 6,
                },
                {
                    'name': str(DummyParams.SM_PIXEL_SIZE),
                    'type': 'float',
                    'value': 114.17,
                    'decimals': 6,
                },
                {
                    'name': str(DummyParams.SM_WAVELENGTH),
                    'type': 'float',
                    'value': 650,
                    'decimals': 6,
                },
                {
                    'name': str(DummyParams.SM_NA),
                    'type': 'float',
                    'value': 1.49,
                    'decimals': 6,
                },
            ],
        }

        # drift simulation
        DRIFT_SIM = {
            'name': str(DummyParams.DRIFT_SIM),
            'type': 'group',
            'expanded': False,
            'children': [
                {
                    'name': str(DummyParams.DRIFT_SPEED),
                    'type': 'float',
                    'value': 0.1,
                },
                {
                    'name': str(DummyParams.DRIFT_PERIOD),
                    'type': 'int',
                    'value': 90,
                },
                {
                    'name': str(DummyParams.DRIFT_MOMENTUM),
                    'type': 'float',
                    'value': 0.8,
                    'limits': [0, 1],
                },
                {
                    'name': str(DummyParams.DRIFT_RANDOM_WALK),
                    'type': 'float',
                    'value': 0.80,
                },
                {
                    'name': str(DummyParams.DRIFT_VIBRATION),
                    'type': 'float',
                    'value': 0.1,
                },
                {
                    'name': str(DummyParams.DRIFT_SIGMA_X),
                    'type': 'float',
                    'value': 5,
                },
                {
                    'name': str(DummyParams.DRIFT_SIGMA_Y),
                    'type': 'float',
                    'value': 20,
                },
                {
                    'name': str(DummyParams.DRIFT_AMPLITUDE),
                    'type': 'float',
                    'value': 5000,
                }
            ],
        }

        # add parameters
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, HEIGHT)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, WIDTH)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, BINNING_HORIZONTAL)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, BINNING_VERTICAL)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, BIT_DEPTH)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, GAIN)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, FULL_WELL_CAPACITY)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, QUANTUM_EFFICIENCY)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, DARK_CURRENT)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, READOUT_NOISE)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, NOISE_BASELINE)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, FLUX)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, PATTERN_TYPE)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, PATTERN_OFFSET)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, PATTERN_SINUSOIDAL)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, DRIFT_SIM)
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, SM_SIM)

        # update params
        self.camera_options.paramsChanged.connect(self.update_cam)

        # start freerun mode button
        freerun = self.get_event_action(DummyParams.FREERUN)
        self.camera_options.add_param_child(CamParams.ACQUISITION, freerun)
        self.camera_options.get_param(DummyParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        stop = {'name': str(DummyParams.STOP), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQUISITION, stop)
        self.camera_options.get_param(DummyParams.STOP).sigActivated.connect(self.stop)

        # config buttons
        load = {'name': str(DummyParams.LOAD), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, load)
        self.camera_options.get_param(DummyParams.LOAD).sigActivated.connect(
            self.load_config
        )

        save = {'name': str(DummyParams.SAVE), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, save)
        self.camera_options.get_param(DummyParams.SAVE).sigActivated.connect(
            self.save_config
        )

        # ROI
        self.camera_options.set_roi_limits(
            (0, self.cam.getWidth()),
            (0, self.cam.getHeight()),
            (32, self.cam.getWidth()),
            (32, self.cam.getHeight()),
        )

    @property
    def cam(self):
        '''The miDummy property.

        Returns
        -------
        miDummy
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: miDummy):
        '''The miDummy property.

        Parameters
        ----------
        cam : miDummy
            the miDummy to set as panel camera.
        '''
        self._cam = cam

    def update_cam(self, param, param_value):
        '''Update the camera parameters.'''
        path = self.camera_options.param_tree.childPath(param)

        if path is None:
            return

        try:
            param_name = DummyParams('.'.join(path))
        except ValueError:
            return

        if param_name == DummyParams.HEIGHT:
            self._cam.height = param_value
        elif param_name == DummyParams.WIDTH:
            self._cam.width = param_value
        elif param_name == DummyParams.BINNING_HORIZONTAL:
            self._cam.binning_horizontal = param_value
        elif param_name == DummyParams.BINNING_VERTICAL:
            self._cam.binning_vertical = param_value
        elif param_name == DummyParams.BIT_DEPTH:
            self._cam.bit_depth = param_value
        elif param_name == DummyParams.GAIN:
            self._cam.gain = param_value
        elif param_name == DummyParams.FULL_WELL_CAPACITY:
            self._cam.full_well_capacity = param_value
        elif param_name == DummyParams.QUANTUM_EFFICIENCY:
            self._cam.quantum_efficiency = param_value
        elif param_name == DummyParams.DARK_CURRENT:
            self._cam.dark_current = param_value
        elif param_name == DummyParams.READOUT_NOISE:
            self._cam.readout_noise = param_value
        elif param_name == DummyParams.NOISE_BASELINE:
            self._cam.noise_baseline = param_value
        elif param_name == DummyParams.FLUX:
            self._cam.flux = param_value
        elif param_name == DummyParams.PATTERN_TYPE:
            self._cam.pattern_type = param_value
        elif param_name == DummyParams.PATTERN_OFFSET:
            self._cam.pattern_offset = param_value

        elif param_name == DummyParams.SINUSOIDAL_AMPLITUDE:
            self._cam.sinusoidal_amplitude = param_value
        elif param_name == DummyParams.SINUSOIDAL_FREQUENCY:
            self._cam.sinusoidal_frequency = param_value
        elif param_name == DummyParams.SINUSOIDAL_PHASE:
            self._cam.sinusoidal_phase = param_value
        elif param_name == DummyParams.SINUSOIDAL_DIRECTION:
            self._cam.sinusoidal_direction = param_value

        elif param_name == DummyParams.SM_INTENSITY:
            self._cam.sm_intensity = param_value
        elif param_name == DummyParams.SM_DENSITY:
            self._cam.sm_density = param_value
        elif param_name == DummyParams.SM_PIXEL_SIZE:
            self._cam.sm_pixel_size = param_value
        elif param_name == DummyParams.SM_WAVELENGTH:
            self._cam.sm_wavelength = param_value
        elif param_name == DummyParams.SM_NA:
            self._cam.sm_na = param_value

        elif param_name == DummyParams.DRIFT_SPEED:
            self._cam.drift_speed = param_value
        elif param_name == DummyParams.DRIFT_PERIOD:
            self._cam.drift_period = param_value
        elif param_name == DummyParams.DRIFT_MOMENTUM:
            self._cam.drift_momentum = param_value
        elif param_name == DummyParams.DRIFT_RANDOM_WALK:
            self._cam.drift_random_walk_factor = param_value
        elif param_name == DummyParams.DRIFT_VIBRATION:
            self._cam.drift_vibration_amplitude = param_value
        elif param_name == DummyParams.DRIFT_SIGMA_X:
            self._cam.drift_sigma_x = param_value
        elif param_name == DummyParams.DRIFT_SIGMA_Y:
            self._cam.drift_sigma_y = param_value
        elif param_name == DummyParams.DRIFT_AMPLITUDE:
            self._cam.drift_amplitude = param_value

    def set_ROI(self):
        '''Sets the ROI for the slected miDummy'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        self.cam.set_roi(*self.camera_options.get_roi_info(False))

    def reset_ROI(self):
        '''Resets the ROI for the slected miDummy'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot reset ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        self.cam.reset_roi()

        self.camera_options.set_roi_info(
            0, 0, int(self.cam.width), int(self.cam.height)
        )

    def center_ROI(self):
        '''Sets the ROI for the slected miDummy'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot center ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        x, y, width, height = self.camera_options.get_roi_info(False)
        x = (int(self.cam.width) - width) // 2
        y = (int(self.cam.height) - height) // 2

        self.cam.set_roi(x, y, width, height)

        self.camera_options.set_roi_info(x, y, width, height)

    def select_ROI(self):
        '''
        Opens a dialog to select a ROI from the last image.
        '''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        if self.acq_job is not None:
            try:

                def work_func(**kwargs):
                    try:
                        image = uImage(self.acq_job.frame.image)

                        image.equalizeLUT(nLUT=True)

                        scale_factor = get_scaling_factor(image.height, image.width)

                        selector = MultiRectangularROISelector.get_selector(
                            image._view, scale_factor, max_rois=1
                        )
                        # if old_rois:
                        #     selector.rois = old_rois

                        rois = selector.select_rectangular_rois()

                        rois = convert_rois_to_pos_size(rois)

                        if len(rois) > 0:
                            return rois[0]
                        else:
                            return None
                    except Exception:
                        traceback.print_exc()
                        return None

                def done(result: list):
                    if result is not None:
                        # x, y, w, h = result
                        self.camera_options.set_roi_info(*result)

                self.worker = QThreadWorker(work_func)
                self.worker.signals.result.connect(done)
                # Execute
                self._threadpool.start(self.worker)
            except Exception:
                traceback.print_exc()

    def exposure_spin_changed(self, param, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in milliseconds
        '''
        self._cam.exposure = value

        self.refresh_exposure()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, float(self._cam.exposure_current)
        )

    def refresh_exposure(self):
        self.camera_options.set_param_value(
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_spin_changed
        )

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        cam : miDummy
            the miDummy used to acquire frames.
        '''
        try:
            # Continuous image capture
            while self.acq_job.frames_captured < self.acq_job.frames or self.mini:
                cycle_time = time.perf_counter_ns()

                self._buffer.put(
                    self._cam.get_dummy_image_from_pattern(
                        self.acq_job.frames_captured / 100
                    ).tobytes()
                )
                # add sensor temperature to the stack
                self._temps.put(self.cam.get_temperature())
                self.acq_job.frames_captured += 1

                diff = (
                    int(self._cam.exposure * 1000)
                    - (time.perf_counter_ns() - cycle_time) // 1000
                )
                if diff > 0:
                    QtCore.QThread.usleep(diff)
                else:
                    QtCore.QThread.usleep(100)  # sleep 100us

                if self.acq_job.stop_threads:
                    break  # in case stop threads is initiated

            self.acq_job.stop_threads = True
            logging.debug('Stop')
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            self._cam.acquisition = False
            QtCore.QThreadPool.globalInstance().releaseThread()
        return QDateTime.currentDateTime()

    def getCaptureArgs(self) -> list:
        '''User specific arguments to be passed to the parallelized
        Camera_Panel.cam_capture function.

        Returns
        -------
        list
            list of args to be passed in order to
            Camera_Panel.cam_capture function.

        Raises
        ------
        NotImplementedError
            Has to be implemented by the use in child class.
        '''
        args = []
        return args

    def save_config(self):
        filename, _ = getSaveFileName(self, 'Save config', filter='XML Files (*.xml);;')

        if len(filename) > 0:
            with self.cam.cam:
                self.cam.cam.save_settings(filename, vb.PersistType.All)

            QtWidgets.QMessageBox.information(self, 'Info', 'Config saved.')
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Config not saved.')

    def load_config(self):
        filename, _ = getOpenFileName(self, 'Load config', filter='XML Files (*.xml);;')

        if len(filename) > 0:
            with self.cam.cam:
                self.cam.default()

                # Load camera settings from file.
                self.cam.cam.load_settings(filename, vb.PersistType.All)

                w, h, x, y = self.cam.get_roi()

                self.camera_options.set_roi_info(int(x), int(y), int(w), int(h))

                self.cam_trigger_mode_cbox.setCurrentText(self.cam.get_trigger_mode())

                self.camera_options.set_param_value(
                    CamParams.EXPOSURE, self.cam.get_exposure()
                )
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No file selected.')
