import json
import logging
import traceback

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.peak.peak_camera import (
    TARGET_PIXEL_FORMAT,
    PeakCamera,
    ids_peak_ipl_extension,
    peakParams,
)
from microEye.qt import QDateTime, QtCore, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.metadata_tree import MetaParams


class PeakPanel(Camera_Panel):
    '''
    A Qt Widget for controlling an IDS Peak / uEye+ Camera | Inherits Camera_Panel
    '''

    PARAMS = peakParams

    FACTOR = 1e3  # exposure time factor to convert to ms

    def __init__(self, cam: PeakCamera, mini=False, *args, **kwargs):
        '''
        Initializes a new PeakPanel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        cam : PeakCamera
            IDS Peak uEye+ Camera python adapter.

        mini : bool, optional
            Flag indicating if this is a mini camera panel, by default False.

        Other Parameters
        ---------------
        *args
            Arguments to pass to the Camera_Panel constructor.

        **kwargs
            Keyword arguments to pass to the Camera_Panel constructor.
        '''
        super().__init__(cam, mini, *args, **kwargs)

    def _init_camera_specific(self):
        # flag true to close camera adapter and dispose it
        self._dispose_cam = False

        for prop in self.cam.property_tree():
            self.camera_options.add_param_child(
                prop.pop('parent', CamParams.ACQ_SETTINGS), prop
            )

        # pixel clock label and combobox
        self.camera_options.get_param(peakParams.CLOCK_FREQ).sigValueChanged.connect(
            lambda: self.cam_pixel_cbox_changed(peakParams.CLOCK_FREQ)
        )

        # framerate slider control
        self.camera_options.get_param(peakParams.FRAMERATE).sigValueChanged.connect(
            self.framerate_changed
        )

        # exposure text box
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits(
            (
                self._cam.get_exposure_range(False)[0],
                self._cam.get_exposure_range(False)[1],
            )
        )
        exposure.setOpts(suffix=self._cam.get_exposure_unit())
        exposure.setValue(self._cam.get_exposure(False))
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        # trigger combobox
        self.camera_options.get_param(
            peakParams.TRIGGER_SOURCE
        ).sigValueChanged.connect(lambda: self.trigger_source_changed())
        self.camera_options.get_param(peakParams.TRIGGER_MODE).sigValueChanged.connect(
            lambda: self.trigger_mode_changed()
        )

        # flash mode combobox
        flash_mode = self.camera_options.get_param(peakParams.FLASH_REF)
        flash_mode.setOpts(enabled=not self.mini)
        flash_mode.sigValueChanged.connect(lambda: self.flash_reference_changed())

        # flash duration slider
        falsh_duration = self.camera_options.get_param(peakParams.FLASH_DURATION)
        falsh_duration.setOpts(enabled=not self.mini)
        falsh_duration.sigValueChanged.connect(self.flash_duration_changed)

        # flash delay
        flash_delay = self.camera_options.get_param(peakParams.FLASH_START_DELAY)
        flash_delay.setOpts(enabled=not self.mini)
        flash_delay.sigValueChanged.connect(self.flash_start_delay_changed)

        # start freerun mode
        self.camera_options.get_param(peakParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # start freerun mode
        self.camera_options.get_param(peakParams.SOFTWARE_TRIGGER).sigActivated.connect(
            lambda: self._cam.software_trigger()
        )

        # stop acquisition
        self.camera_options.get_param(peakParams.STOP).sigActivated.connect(
            lambda: self.stop()
        )

        # ROI
        self.camera_options.set_roi_limits(
            (0, self._cam.node_map.FindNode('SensorWidth').Value()),
            (0, self._cam.node_map.FindNode('SensorHeight').Value()),
            (
                self._cam.node_map.FindNode('Width').Minimum(),
                self._cam.node_map.FindNode('SensorWidth').Value(),
            ),
            (
                self._cam.node_map.FindNode('Height').Minimum(),
                self._cam.node_map.FindNode('SensorHeight').Value(),
            ),
        )

    @property
    def cam(self):
        '''The PeakCamera property.

        Returns
        -------
        PeakCamera
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: PeakCamera):
        '''The PeakCamera property.

        Parameters
        ----------
        cam : PeakCamera
            the PeakCamera to set as panel camera.
        '''
        self._cam = cam

    def refresh_clock_freq(self):
        clock_freq = self.camera_options.get_param(peakParams.CLOCK_FREQ)

        clock_freq.blockSignals(True)
        clock_freq.setValue(self.cam.get_device_clock_frequency())
        clock_freq.blockSignals(False)

    def set_ROI(self):
        '''Sets the ROI for the slected PeakCamera'''
        super().set_ROI()

        self.refresh_clock_freq()

    def reset_ROI(self):
        '''Resets the ROI for the slected PeakCamera'''
        super().reset_ROI()

        self.refresh_clock_freq()

    def trigger_source_changed(self):
        '''
        Slot for changed trigger source
        '''
        source = self.camera_options.get_param_value(peakParams.TRIGGER_SOURCE)
        self._cam.set_trigger_source(source)
        self._cam.get_trigger_source()

        mode = self.camera_options.get_param(peakParams.TRIGGER_MODE)
        mode.blockSignals(True)
        mode.setValue(self._cam.get_trigger_mode())
        mode.blockSignals(False)
        self._cam.get_trigger_source(output=True)

    def trigger_mode_changed(self):
        '''
        Slot for changed trigger mode
        '''
        mode = self.camera_options.get_param_value(peakParams.TRIGGER_MODE)
        self._cam.set_trigger_mode(mode)
        self._cam.get_trigger_mode()

    def flash_reference_changed(self):
        '''
        Slot for changed flash mode

        Parameters
        ----------
        param : peakParams
            parameter path to index of selected flash mode from PeakCamera.FLASH_MODES
        '''
        ref = self.camera_options.get_param_value(peakParams.FLASH_REF)
        self._cam.set_flash_reference(ref)
        self._cam.get_flash_reference(output=True)

    def cam_pixel_cbox_changed(self, param: peakParams):
        '''
        Slot for changed pixel clock

        Parameters
        ----------
        param : peakParams
            parameter path Enum value pointing to pixel clock parameter in MHz
            (note that only certain values are allowed by camera)
        '''
        value = self.camera_options.get_param_value(param)
        self._cam.set_device_clock_frequency(int(value))

        self.camera_options.set_param_value(
            peakParams.CLOCK_FREQ,
            self._cam.get_device_clock_frequency(False),
            self.cam_pixel_cbox_changed,
        )

        self.refresh_framerate()

    def framerate_changed(self, param, value):
        '''
        Slot for changed framerate

        Parameters
        ----------
        param : Parameter
            frames per second parameter
        Value : double
            selected frames per second
        '''
        self._cam.set_framerate(value)
        # self._cam.get_framerate(False)

        self.camera_options.set_param_value(
            peakParams.FRAMERATE,
            self._cam.get_framerate(False),
            self.framerate_changed,
        )

        self.refresh_exposure()

    def exposure_spin_changed(self, param, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        param : Parameter
            exposure parameter
        Value : double
            selected exposure in milliseconds
        '''
        self._cam.set_exposure(value)

        self.camera_options.set_param_value(
            CamParams.EXPOSURE,
            self._cam.get_exposure(False),
            self.exposure_spin_changed,
        )

        self.refresh_flash()

        self.OME_tab.set_param_value(MetaParams.EXPOSURE, self._cam.get_exposure(False))
        if self.master:
            self.exposureChanged.emit()

    def refresh_framerate(self, value=None):
        self._cam.get_framerate_range(False)

        framerate = self.camera_options.get_param(peakParams.FRAMERATE)
        framerate.setLimits(
            (
                self._cam.get_framerate_range(False)[0],
                self._cam.get_framerate_range(False)[1],
            )
        )

        self.framerate_changed(framerate, self._cam.get_framerate_range(False)[1])

    def refresh_exposure(self):
        self._cam.get_exposure_range(False)

        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits(
            (
                self._cam.get_exposure_range(False)[0],
                self._cam.get_exposure_range(False)[1],
            )
        )
        exposure.setValue(self._cam.get_exposure_range(False)[1])

    def refresh_flash(self):
        duration = self.camera_options.get_param(peakParams.FLASH_DURATION)
        duration.setLimits((0, self._cam.get_flash_duration_range(False)[1]))
        duration.setValue(0)

        delay = self.camera_options.get_param(peakParams.FLASH_DURATION)
        delay.setLimits((0, self._cam.get_flash_delay_range(False)[1]))
        delay.setValue(0)

    def flash_duration_changed(self, param, value):
        '''
        Slot for changed flash duration

        Parameters
        ----------
        param : Parameter
            flash duration parameter
        Value : double
            selected flash duration in micro-seconds
        '''
        self._cam.set_flash_duration(value)

        self.camera_options.set_param_value(
            peakParams.FLASH_DURATION,
            self._cam.get_flash_duration(False),
            self.flash_duration_changed,
        )

    def flash_start_delay_changed(self, param, value):
        '''
        Slot for changed flash delay

        Parameters
        ----------
        param : Parameter
            flash delay parameter
        Value : double
            selected flash delay in micro-seconds
        '''
        self._cam.set_flash_delay(value)

        self.camera_options.set_param_value(
            peakParams.FLASH_START_DELAY,
            self._cam.get_flash_delay(False),
            self.flash_start_delay_changed,
        )

    def start_free_run(self, param=None, Prefix='', Event=None):
        '''
        Starts free run acquisition mode

        Parameters
        ----------
        param : Parameter
            the parameter that was activated.
        Prefix : str
            an extra prefix added to the image stack file name.
        '''
        if self._cam.acquisition:
            return  # if acquisition is already going on

        self._cam.refresh_info()  # refresh adapter info

        self._save_prefix = Prefix
        self.acq_job = self.getAcquisitionJob(Event=Event)

        ret = self._cam.start_acquisition()

        # start both capture and display workers
        if ret:
            self.start_all_workers()

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        nRet : int
            return code from PeakCamera, ueye.IS_SUCCESS = 0 to run.
        cam : PeakCamera
            the PeakCamera used to acquire frames.
        '''
        try:
            time = QDateTime.currentDateTime()
            # Continuous image capture
            while self.acq_job.c_event.is_set() is False:
                self.acq_job.capture_time = time.msecsTo(QDateTime.currentDateTime())
                time = QDateTime.currentDateTime()

                try:
                    # Wait until the next buffer is received.
                    buffer = self._cam._datastream.WaitForFinishedBuffer(1000)
                except Exception:
                    buffer = None
                    logging.getLogger(__name__).error(
                        'Exception in WaitForFinishedBuffer',
                        exc_info=True,
                    )

                if buffer is not None:
                    # Get image from buffer (shallow copy)
                    self.ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)

                    self._cam._datastream.QueueBuffer(buffer)

                    self._buffer.put(self.ipl_image.get_numpy().copy())
                    # add sensor temperature to the stack
                    self._temps.put(self._cam.get_temperature())
                    self.acq_job.frames_captured = self.acq_job.frames_captured + 1
                    if (
                        self.acq_job.frames_captured >= self.acq_job.frames
                        and not self.mini
                    ):
                        self.acq_job.c_event.set()
                        self.acq_job.stop_threads = True
                        logging.getLogger(__name__).debug('Stop')

                QtCore.QThread.usleep(100)  # sleep 100us
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            self._cam.stop_acquisition()
            self._cam.acquisition = False
            self._threadpool.releaseThread()

    def getCaptureArgs(self) -> list:
        '''User specific arguments to be passed to the parallelized
        Camera_Panel.cam_capture function.

        Example
        ------
        check child panels (vimba, ueye or thorlabs)

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

    def dispose(self):
        self._cam._device = None
        self._cam._datastream = None
        self._cam._buffer_list = []
        self._cam._image_converter = None

        self._cam = None
