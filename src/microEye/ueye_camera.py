from ctypes import *
import logging
import numpy as np

try:
    from pyueye import ueye
except Exception:
    ueye = None

formats_ = {ueye.IS_CM_SENSOR_RAW8: 8,
            ueye.IS_CM_SENSOR_RAW10: 16,
            ueye.IS_CM_SENSOR_RAW12: 16,
            ueye.IS_CM_SENSOR_RAW16: 16,

            ueye.IS_CM_MONO8: 8,
            ueye.IS_CM_MONO10: 16,
            ueye.IS_CM_MONO12: 16,
            ueye.IS_CM_MONO16: 16,

            ueye.IS_CM_RGB8_PLANAR: 24,
            ueye.IS_CM_RGB8_PACKED: 24,
            ueye.IS_CM_RGBA8_PACKED: 32,
            ueye.IS_CM_RGBY8_PACKED: 32,
            ueye.IS_CM_RGB10_PACKED: 32,

            ueye.IS_CM_RGB10_UNPACKED: 48,
            ueye.IS_CM_RGB12_UNPACKED: 48,
            ueye.IS_CM_RGBA12_UNPACKED: 64,

            ueye.IS_CM_BGR5_PACKED: 16,
            ueye.IS_CM_BGR565_PACKED: 16,
            ueye.IS_CM_BGR8_PACKED: 24,
            ueye.IS_CM_BGRA8_PACKED: 32,
            ueye.IS_CM_BGRY8_PACKED: 32,
            ueye.IS_CM_BGR10_PACKED: 32,

            ueye.IS_CM_BGR10_UNPACKED: 48,
            ueye.IS_CM_BGR12_UNPACKED: 48,
            ueye.IS_CM_BGRA12_UNPACKED: 64,

            ueye.IS_CM_UYVY_PACKED: 16,
            ueye.IS_CM_UYVY_MONO_PACKED: 16,
            ueye.IS_CM_UYVY_BAYER_PACKED: 16,
            ueye.IS_CM_CBYCRY_PACKED: 16
            }

formats_strs = {ueye.IS_CM_SENSOR_RAW8: "IS_CM_SENSOR_RAW8",
                ueye.IS_CM_SENSOR_RAW10: "IS_CM_SENSOR_RAW10",
                ueye.IS_CM_SENSOR_RAW12: "IS_CM_SENSOR_RAW12",
                ueye.IS_CM_SENSOR_RAW16: "IS_CM_SENSOR_RAW16",

                ueye.IS_CM_MONO8: "IS_CM_MONO8",
                ueye.IS_CM_MONO10: "IS_CM_MONO10",
                ueye.IS_CM_MONO12: "IS_CM_MONO12",
                ueye.IS_CM_MONO16: "IS_CM_MONO16",

                ueye.IS_CM_RGB8_PLANAR: "IS_CM_RGB8_PLANAR",
                ueye.IS_CM_RGB8_PACKED: "IS_CM_RGB8_PACKED",
                ueye.IS_CM_RGBA8_PACKED: "IS_CM_RGB8_PACKED",
                ueye.IS_CM_RGBY8_PACKED: "IS_CM_RGBY8_PACKED",
                ueye.IS_CM_RGB10_PACKED: "IS_CM_RGB10_PACKED",

                ueye.IS_CM_RGB10_UNPACKED: "IS_CM_RGB10_UNPACKED",
                ueye.IS_CM_RGB12_UNPACKED: "IS_CM_RGB12_UNPACKED",
                ueye.IS_CM_RGBA12_UNPACKED: "IS_CM_RGBA12_UNPACKED",

                ueye.IS_CM_BGR5_PACKED: "IS_CM_BGR5_PACKED",
                ueye.IS_CM_BGR565_PACKED: "IS_CM_BGR565_PACKED",
                ueye.IS_CM_BGR8_PACKED: "IS_CM_BGR8_PACKED",
                ueye.IS_CM_BGRA8_PACKED: "IS_CM_BGRA8_PACKED",
                ueye.IS_CM_BGRY8_PACKED: "IS_CM_BGRY8_PACKED",
                ueye.IS_CM_BGR10_PACKED: "IS_CM_BGR10_PACKED",

                ueye.IS_CM_BGR10_UNPACKED: "IS_CM_BGR10_UNPACKED",
                ueye.IS_CM_BGR12_UNPACKED: "IS_CM_BGR12_UNPACKED",
                ueye.IS_CM_BGRA12_UNPACKED: "IS_CM_BGRA12_UNPACKED",

                ueye.IS_CM_UYVY_PACKED: "IS_CM_UYVY_PACKED",
                ueye.IS_CM_UYVY_MONO_PACKED: "IS_CM_UYVY_MONO_PACKED",
                ueye.IS_CM_UYVY_BAYER_PACKED: "IS_CM_UYVY_BAYER_PACKED",
                ueye.IS_CM_CBYCRY_PACKED: "IS_CM_CBYCRY_PACKED"
                }

color_modes_ = {
    ueye.IS_COLORMODE_INVALID: 'IS_COLORMODE_INVALID',
    ueye.IS_COLORMODE_MONOCHROME: 'IS_COLORMODE_MONOCHROME',
    ueye.IS_COLORMODE_BAYER: 'IS_COLORMODE_BAYER',
    ueye.IS_COLORMODE_CBYCRY: 'IS_COLORMODE_CBYCRY',
    ueye.IS_COLORMODE_JPEG: 'IS_COLORMODE_JPEG'}


class IDS_Camera:
    '''A class to handle an IDS uEye camera.'''

    TRIGGER_MODES = {
        "Trigger Off": ueye.IS_SET_TRIGGER_OFF,
        "Software Trigger": ueye.IS_SET_TRIGGER_SOFTWARE,
        "Falling edge external trigger": ueye.IS_SET_TRIGGER_HI_LO,
        "Rising edge external trigger": ueye.IS_SET_TRIGGER_LO_HI}
    '''Tigger modes supported by IDS uEye cameras.

    Returns
    -------
    dict[str, int]
        dictionary used for GUI display and control.
    '''

    FLASH_MODES = {
        "Flash Off": ueye.IO_FLASH_MODE_OFF,
        "Flash Trigger Low Active": ueye.IO_FLASH_MODE_TRIGGER_LO_ACTIVE,
        "Flash Trigger High Active": ueye.IO_FLASH_MODE_TRIGGER_HI_ACTIVE,
        "Flash Constant High": ueye.IO_FLASH_MODE_CONSTANT_HIGH,
        "Flash Constant Low": ueye.IO_FLASH_MODE_CONSTANT_LOW,
        "Flash Freerun Low Active": ueye.IO_FLASH_MODE_FREERUN_LO_ACTIVE,
        "Flash Freerun High Active": ueye.IO_FLASH_MODE_FREERUN_HI_ACTIVE}
    '''Flash modes supported by IDS uEye cameras.

    Returns
    -------
    dict[str, int]
        dictionary used for GUI display and control.
    '''

    def __init__(self, Cam_ID=0):
        '''Initializes a new IDS_Camera with the specified Cam_ID.


        Parameters
        ----------
        Cam_ID : int, optional
            camera ID, by default 0

            0: first available camera

            1-254: The camera with the specified camera ID
        '''
        # Variables
        self.Cam_ID = Cam_ID
        # 0: first available camera;
        # 1-254: The camera with the specified camera ID
        self.hCam = ueye.HIDS(Cam_ID)
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.dInfo = ueye.IS_DEVICE_INFO()
        self.rectAOI = ueye.IS_RECT()
        self.set_rectAOI = ueye.IS_RECT()
        self.minAOI = ueye.IS_SIZE_2D()

        self.pitch = ueye.INT()
        self.MemInfo = []
        self.current_buffer = ueye.c_mem_p()
        self.current_id = ueye.int()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()

        self.bit_depth = ueye.INT(24)
        # 3: channels for color mode(RGB);
        # take 1 channel for monochrome
        self.channels = 1
        self.color_mode = ueye.INT()
        self.bit_depth = ueye.UINT(0)
        self.supported_bit_depth = ueye.UINT(0)
        self.bytes_per_pixel = int(self.bit_depth / 8)

        self.pixel_clock = ueye.c_uint(0)
        self.pixel_clock_default = ueye.c_uint()
        self.pixel_clock_list = (ueye.c_uint * 5)()
        self.pixel_clock_range = (ueye.c_uint * 3)()
        self.pixel_clock_count = ueye.c_uint()

        self.minFrameRate = ueye.c_double(0)
        self.maxFrameRate = ueye.c_double(0)
        self.incFrameRate = ueye.c_double(0)
        self.currentFrameRate = ueye.c_double(0)
        self.exposure_range = (ueye.c_double * 3)()
        self.exposure_current = ueye.c_double(0)

        self.flash_mode = ueye.c_uint(ueye.IO_FLASH_MODE_OFF)
        self.flash_min = ueye.IO_FLASH_PARAMS()
        self.flash_max = ueye.IO_FLASH_PARAMS()
        self.flash_inc = ueye.IO_FLASH_PARAMS()
        self.flash_cur = ueye.IO_FLASH_PARAMS()

        self.CaptureStatusInfo = ueye.UEYE_CAPTURE_STATUS_INFO()

        self.acquisition = False
        self.capture_video = False
        self.memory_allocated = False
        self.trigger_mode = None
        self.temperature = -127

    def initialize(self):
        '''Starts the driver and establishes the connection to the camera'''
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera " + str(self.Cam_ID) + " ERROR")

        self.get_coded_info()
        self.get_sensor_info()
        self.resetToDefault()
        self.setDisplayMode()
        self.setColorMode()
        self.get_AOI()
        self.print_cam_info()

        self.name = self.sInfo.strSensorName.decode('utf-8') + "_" + \
            self.cInfo.SerNo.decode('utf-8')
        self.name = str.replace(self.name, "-", "_")

        self.refresh_info()

    def refresh_info(self, output=False):
        '''Refreshes the camera's pixel clock,
        framerate, exposure and flash ranges.

        Parameters
        ----------
        output : bool, optional
            True to printout errors, by default False
        '''
        if output:
            print(self.name)
            print()
        self.get_pixel_clock_info(output)
        self.get_framerate_range(output)
        self.get_exposure_range(output)
        self.get_exposure(output)
        self.get_flash_range(output)

    @staticmethod
    def get_camera_list(output=False):
        '''Gets the list of available IDS cameras

        Parameters
        ----------
        output : bool, optional
            print out camera list, by default False

        Returns
        -------
        dict[str, object]
            dictionary containing
            {camID, devID, senID, Status, InUse, Model, Serial}
        '''
        cam_list = []
        cam_count = ueye.c_int(0)
        nRet = ueye.is_GetNumberOfCameras(cam_count)
        if nRet == ueye.IS_SUCCESS and cam_count > 0:
            pucl = ueye.UEYE_CAMERA_LIST(
                ueye.UEYE_CAMERA_INFO * cam_count.value)
            pucl.dwCount = cam_count.value
            if (ueye.is_GetCameraList(pucl) == ueye.IS_SUCCESS):
                for index in range(cam_count.value):
                    cam_list.append({
                        "camID": pucl.uci[index].dwCameraID.value,
                        "devID": pucl.uci[index].dwDeviceID.value,
                        "senID": pucl.uci[index].dwSensorID.value,
                        "Status": pucl.uci[index].dwStatus.value,
                        "InUse": pucl.uci[index].dwInUse.value,
                        "Model": pucl.uci[index].Model.decode('utf-8'),
                        "Serial": pucl.uci[index].SerNo.decode('utf-8'),
                        "Driver": 'uEye'})
                if output:
                    print(cam_list)
        return cam_list

    def get_coded_info(self):
        '''Reads out the data hard-coded in the non-volatile camera memory
        and writes it to the data structure that cInfo points to
        '''
        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

    def get_device_info(self):
        '''Gets the device info, used to get the sensor temp.
        '''
        nRet = ueye.is_DeviceInfo(
            self.hCam,
            ueye.IS_DEVICE_INFO_CMD_GET_DEVICE_INFO,
            self.dInfo,
            ueye.sizeof(self.dInfo))

        if nRet != ueye.IS_SUCCESS:
            print("is_DeviceInfo ERROR")

    def get_temperature(self):
        '''Reads out the sensor temperature value

        Returns
        -------
        float
            camera sensor temperature in C.
        '''
        self.get_device_info()
        self.temperature = ((
            (self.dInfo.infoDevHeartbeat.wTemperature.value >> 4) & 127) * 1.0
            + ((self.dInfo.infoDevHeartbeat.wTemperature.value & 15) / 10.0))
        return self.temperature

    def get_sensor_info(self):
        '''You can query additional information about
        the sensor type used in the camera
        '''
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

    def resetToDefault(self):
        '''Resets all parameters to the camera-specific defaults
        as specified by the driver.

        By default, the camera uses full resolution, a medium speed
        and color level gain values adapted to daylight exposure.
        '''
        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")

    def setDisplayMode(self, mode=ueye.IS_SET_DM_DIB):
        '''Captures an image in system memory (RAM).

        Using is_RenderBitmap(), you can define the image display (default).

        Parameters
        ----------
        mode : [type], optional
            Capture mode, by default ueye.IS_SET_DM_DIB
            captures an image in system memory (RAM).

        Returns
        -------
        int
            is_SetDisplayMode return code
        '''
        nRet = ueye.is_SetDisplayMode(self.hCam, mode)
        return nRet

    def setColorMode(self):
        '''Set the right color mode, by camera default.
        '''
        # self.color_mode = ueye.is_SetColorMode(
        #     self.hCam, ueye.IS_GET_COLOR_MODE)
        nCmode = int.from_bytes(self.sInfo.nColorMode.value, byteorder='big')
        nRet = ueye.is_DeviceFeature(
            self.hCam,
            ueye.IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_SENSOR_BIT_DEPTHS,
            self.supported_bit_depth, sizeof(self.supported_bit_depth))
        print('nRet', nRet)
        if nCmode == \
                ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(
                self.hCam,
                self.bit_depth,
                self.color_mode)
            self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            print(color_modes_[nCmode], ": ", )
            print(
                "\tcolor_mode: \t\t", formats_strs[self.color_mode.value])
            print("\tbit_depth: \t\t", self.bit_depth)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif nCmode ==\
                ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.color_mode = ueye.IS_CM_BGRA8_PACKED
            self.bit_depth = ueye.INT(32)
            self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            print(color_modes_[nCmode], ": ", )
            print("\tcolor_mode: \t\t", formats_strs[self.color_mode])
            print("\tbit_depth: \t\t", self.bit_depth)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif nCmode ==\
                ueye.IS_COLORMODE_MONOCHROME:
            # for mono camera models that uses CM_MONO12 mode
            print('Sbit', self.supported_bit_depth)
            if nRet == ueye.IS_SUCCESS:
                if (self.supported_bit_depth and
                        ueye.IS_SENSOR_BIT_DEPTH_12_BIT) != 0:
                    self.color_mode = ueye.IS_CM_MONO12
                    self.bit_depth = ueye.INT(12)
                    self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
                elif (self.supported_bit_depth and
                        ueye.IS_SENSOR_BIT_DEPTH_10_BIT) != 0:
                    self.color_mode = ueye.IS_CM_MONO10
                    self.bit_depth = ueye.INT(10)
                    self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
                elif (self.supported_bit_depth and
                        ueye.IS_SENSOR_BIT_DEPTH_8_BIT) != 0:
                    self.color_mode = ueye.IS_CM_MONO8
                    self.bit_depth = ueye.INT(8)
                    self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            else:
                self.color_mode = ueye.IS_CM_MONO8
                self.bit_depth = ueye.INT(8)
                self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))

            print(color_modes_[nCmode], ": ", )
            print("\tcolor_mode: \t\t", formats_strs[self.color_mode])
            print("\tbit_depth: \t\t", self.bit_depth)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            self.color_mode = ueye.IS_CM_MONO8
            self.bit_depth = ueye.INT(8)
            self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            print("Else: ", )
            print("\tcolor_mode: \t\t", formats_strs[self.color_mode])
            print("\tbit_depth: \t\t", self.bit_depth)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)

        nRet = ueye.is_SetColorMode(self.hCam, self.color_mode)

    def get_AOI(self):
        '''Can be used to get the size and position
        of an "area of interest" (AOI) within an image.

        For AOI rectangle check IDS_Camera.rectAOI

        Returns
        -------
        int
            is_AOI return code.
        '''
        nRet = ueye.is_AOI(
            self.hCam,
            ueye.IS_AOI_IMAGE_GET_AOI,
            self.rectAOI,
            ueye.sizeof(self.rectAOI))

        if nRet != ueye.IS_SUCCESS:
            print("is_AOI GET ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height
        return nRet

    def set_AOI(self, x, y, width, height):
        '''Sets the size and position of an
        "area of interest"(AOI) within an image.

        Parameters
        ----------
        x : int
            AOI x position.
        y : int
            AOI y position.
        width : int
            AOI width.
        height : int
            AOI height.

        Returns
        -------
        int
            is_AOI return code.
        '''
        self.set_rectAOI.s32X.value = x
        self.set_rectAOI.s32Y.value = y
        self.set_rectAOI.s32Width.value = width
        self.set_rectAOI.s32Height.value = height

        nRet = ueye.is_AOI(
            self.hCam,
            ueye.IS_AOI_IMAGE_SET_AOI,
            self.set_rectAOI,
            ueye.sizeof(self.set_rectAOI))

        if nRet != ueye.IS_SUCCESS:
            print("is_AOI SET ERROR")

        self.width = self.set_rectAOI.s32Width
        self.height = self.set_rectAOI.s32Height

        return nRet

    def reset_AOI(self):
        '''Resets the AOI.

        Returns
        -------
        int
            is_AOI return code.
        '''
        nRet = ueye.is_AOI(
            self.hCam,
            ueye.IS_AOI_IMAGE_SET_AOI,
            self.rectAOI,
            ueye.sizeof(self.rectAOI))

        if nRet != ueye.IS_SUCCESS:
            print("is_AOI RESET ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        return nRet

    def set_pixel_clock(self, value):
        '''Sets the camera's pixel clock speed.

        Parameters
        ----------
        value : uint
            The supported clock speed to set.

        Returns
        -------
        int
            is_PixelClock return code.
        '''
        set = (ueye.c_uint * 1)(value)
        nRet = ueye.is_PixelClock(
            self.hCam,
            ueye.IS_PIXELCLOCK_CMD_SET,
            set,
            ueye.sizeof(set))

        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock ERROR")

        return nRet

    def get_pixel_clock_info(self, output=True):
        '''Gets the pixel clock value, default,
        and supported values list.

        Parameters
        ----------
        output : bool, optional
            prints out results to terminal, by default True
        '''

        self.get_pixel_clock(output)

        # Get default pixel clock
        nRet = ueye.is_PixelClock(
            self.hCam,
            ueye.IS_PIXELCLOCK_CMD_GET_DEFAULT,
            self.pixel_clock_default,
            ueye.sizeof(self.pixel_clock_default))

        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock Default ERROR")
        elif output:
            print("Default " + str(self.pixel_clock_default.value))

        nRet = ueye.is_PixelClock(
            self.hCam,
            ueye.IS_PIXELCLOCK_CMD_GET_RANGE,
            self.pixel_clock_range,
            ueye.sizeof(self.pixel_clock_range))

        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock Range ERROR")
        elif output:
            print("Min " + str(self.pixel_clock_range[0].value))
            print("Max " + str(self.pixel_clock_range[1].value))

        nRet = ueye.is_PixelClock(
            self.hCam,
            ueye.IS_PIXELCLOCK_CMD_GET_NUMBER,
            self.pixel_clock_count,
            ueye.sizeof(self.pixel_clock_count))

        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock Count ERROR")
        else:
            if output:
                print("Count " + str(self.pixel_clock_count.value))

            self.pixel_clock_list = (ueye.c_uint *
                                     self.pixel_clock_count.value)()

            nRet = ueye.is_PixelClock(
                self.hCam, ueye.IS_PIXELCLOCK_CMD_GET_LIST,
                self.pixel_clock_list,
                self.pixel_clock_count * ueye.sizeof(c_uint))

            if nRet != ueye.IS_SUCCESS:
                print("is_PixelClock List ERROR")
            elif output:
                for clk in self.pixel_clock_list:
                    print("List " + str(clk.value))
                print()

    def get_pixel_clock(self, output=True):
        '''Gets the current pixel clock speed.

        Parameters
        ----------
        output : bool, optional
            print out to terminal, by default True

        Returns
        -------
        int
            pixel clock speed in MHz.
        '''
        nRet = ueye.is_PixelClock(
            self.hCam,
            ueye.IS_PIXELCLOCK_CMD_GET,
            self.pixel_clock,
            ueye.sizeof(self.pixel_clock))

        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock ERROR")
        elif output:
            print("Pixel Clock %d MHz" % self.pixel_clock.value)
        return self.pixel_clock.value

    def get_exposure_range(self, output=True):
        ''' Gets exposure range

        Parameters
        ----------
        output : bool, optional
            [description], by default True
        '''
        nRet = ueye.is_Exposure(
            self.hCam,
            ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE,
            self.exposure_range, ueye.sizeof(self.exposure_range))
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure Range ERROR")
        elif output:
            print("Exposure")
            print("Min " + str(self.exposure_range[0].value))
            print("Max " + str(self.exposure_range[1].value))
            print("Inc " + str(self.exposure_range[2].value))
            print()

    # Get exposure
    def get_exposure(self, output=True):
        nRet = ueye.is_Exposure(
            self.hCam,
            ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
            self.exposure_current, ueye.sizeof(self.exposure_current))
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure Current ERROR")
        elif output:
            print("Current Exposure " + str(self.exposure_current.value))
            print()
        return self.exposure_current.value

    # Set exposure
    def set_exposure(self, value):
        nRet = ueye.is_Exposure(
            self.hCam,
            ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
            ueye.c_double(max(min(value, self.exposure_range[1].value), 0)),
            ueye.sizeof(ueye.c_double))
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure Set ERROR")
        else:
            print("Exposure set to " + str(self.get_exposure(False)))
            print()
        return nRet

    # Get flash output info
    def get_flash_range(self, output=True):
        nRet = ueye.is_IO(
            self.hCam,
            ueye.IS_IO_CMD_FLASH_GET_PARAMS_MIN,
            self.flash_min,
            ueye.sizeof(self.flash_min))
        if nRet != ueye.IS_SUCCESS:
            print("is_IO Flash Min ERROR")
        elif output:
            print("Flash Output")
            print("Min Delay (us) " + str(self.flash_min.s32Delay.value))
            print("Min Duration (us) " + str(self.flash_min.u32Duration.value))
            print()

        nRet = ueye.is_IO(
            self.hCam,
            ueye.IS_IO_CMD_FLASH_GET_PARAMS_MAX,
            self.flash_max,
            ueye.sizeof(self.flash_max))
        if nRet != ueye.IS_SUCCESS:
            print("is_IO Flash Max ERROR")
        elif output:
            print("Max Delay (us) " + str(self.flash_max.s32Delay.value))
            print("Max Duration (us) " + str(self.flash_max.u32Duration.value))
            print()

        nRet = ueye.is_IO(
            self.hCam,
            ueye.IS_IO_CMD_FLASH_GET_PARAMS_INC,
            self.flash_inc,
            ueye.sizeof(self.flash_inc))
        if nRet != ueye.IS_SUCCESS:
            print("is_IO Flash Inc. ERROR")
        elif output:
            print("Inc. Delay (us) " + str(self.flash_inc.s32Delay.value))
            print(
                "Inc. Duration (us) " + str(self.flash_inc.u32Duration.value))
            print()

        self.get_flash_params(output)

    def get_flash_params(self, output=True):
        nRet = ueye.is_IO(
            self.hCam,
            ueye.IS_IO_CMD_FLASH_GET_PARAMS,
            self.flash_cur,
            ueye.sizeof(self.flash_cur))
        if nRet != ueye.IS_SUCCESS:
            print("is_IO Flash Current ERROR")
        elif output:
            print("Current Delay (us) " + str(self.flash_cur.s32Delay.value))
            print("Current Duration (us) " + str(
                self.flash_cur.u32Duration.value))
            print()

    # Set current flash parameters
    def set_flash_params(self, delay, duration):
        params = ueye.IO_FLASH_PARAMS()
        delay = int(delay)
        duration = ueye.UINT(duration)
        if duration != 0:
            params.u32Duration.value = max(
                min(duration, self.flash_max.u32Duration.value),
                self.flash_min.u32Duration.value)
        else:
            params.u32Duration.value = duration
        params.s32Delay.value = max(
            min(delay, self.flash_max.s32Delay.value),
            self.flash_min.s32Delay.value)
        nRet = ueye.is_IO(
            self.hCam,
            ueye.IS_IO_CMD_FLASH_SET_PARAMS,
            params,
            ueye.sizeof(params))
        if nRet != ueye.IS_SUCCESS:
            print("set_flash_params ERROR")
        else:
            self.get_flash_params()
        return nRet

    # Set current flash mode
    def set_flash_mode(self, mode):
        mode = ueye.c_uint(mode)
        nRet = ueye.is_IO(
            self.hCam,
            ueye.IS_IO_CMD_FLASH_SET_MODE,
            mode, ueye.sizeof(mode))
        if nRet != ueye.IS_SUCCESS:
            print("is_IO Set Flash Mode ERROR")
        return nRet

    # Get current flash mode
    def get_flash_mode(self, output=False):
        nRet = ueye.is_IO(
            self.hCam,
            ueye.IS_IO_CMD_FLASH_GET_MODE,
            self.flash_mode, ueye.sizeof(self.flash_mode))
        if nRet != ueye.IS_SUCCESS:
            print("is_IO Get Flash Mode ERROR")
        else:
            print(list(self.FLASH_MODES.keys())[list(
                self.FLASH_MODES.values()).index(self.flash_mode.value)])

    # Get framerate range
    def get_framerate_range(self, output=True):
        nRet = ueye.is_GetFrameTimeRange(
            self.hCam,
            self.minFrameRate,
            self.maxFrameRate,
            self.incFrameRate)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetFrameTimeRange ERROR")
        else:
            temp = self.maxFrameRate.value
            self.maxFrameRate.value = 1/self.minFrameRate.value
            self.minFrameRate.value = 1/temp
            if output:
                print("FrameRate")
                print("Min " + str(self.minFrameRate.value))
                print("Max " + str(self.maxFrameRate.value))
                print()
        return np.array([self.minFrameRate.value, self.maxFrameRate.value])

    # Get current framerate
    def get_framerate(self, output=True):
        nRet = ueye.is_GetFramesPerSecond(self.hCam, self.currentFrameRate)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetFramesPerSecond ERROR")
        elif output:
            print("Current FrameRate " + str(self.currentFrameRate.value))
            print()
        return self.currentFrameRate.value

    # Set current framerate
    def set_framerate(self, value):
        nRet = ueye.is_SetFrameRate(
            self.hCam,
            ueye.c_double(max(
                min(value, self.maxFrameRate.value), self.minFrameRate.value)),
            self.currentFrameRate)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetFrameRate ERROR")
        else:
            print("FrameRate set to " + str(self.currentFrameRate.value))
            print()
        return nRet

    # get trigger mode
    def get_trigger_mode(self):
        nRet = ueye.is_SetExternalTrigger(
            self.hCam,
            ueye.IS_GET_EXTERNALTRIGGER)
        if nRet == ueye.IS_SET_TRIGGER_OFF:
            print("Trigger Off")
        elif nRet == ueye.IS_SET_TRIGGER_SOFTWARE:
            print("Software Trigger")
        elif nRet == ueye.IS_SET_TRIGGER_HI_LO:
            print("Falling edge external trigger")
        elif nRet == ueye.IS_SET_TRIGGER_LO_HI:
            print("Rising  edge external trigger")
        else:
            print("NA")
        self.trigger_mode = nRet
        return nRet

    def set_trigger_mode(self, mode):
        return ueye.is_SetExternalTrigger(self.hCam, mode)

    # Prints out some information about the camera and the sensor
    def print_cam_info(self):
        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t", self.width)
        print("Maximum image height:\t", self.height)
        print()

    # Allocates an image memory for an image having its dimensions defined by
    # width and height and its color depth defined by bit_depth
    def allocate_memory(self):
        if len(self.MemInfo) > 0:
            self.free_memory()

        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.c_int(0)
        self.MemInfo.append([self.pcImageMemory, self.MemID])
        nRet = ueye.is_AllocImageMem(
            self.hCam,
            self.width,
            self.height,
            self.bit_depth,
            self.pcImageMemory,
            self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(
                self.hCam,
                self.pcImageMemory,
                self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                self.memory_allocated = True

    # Activates the camera's live video mode (free run mode)
    def start_live_capture(self):
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")
        else:
            self.capture_video = True
        return nRet

    # Stops the camera's live video mode (free run mode)
    def stop_live_capture(self):
        nRet = ueye.is_StopLiveVideo(self.hCam, ueye.IS_FORCE_VIDEO_STOP)
        if nRet != ueye.IS_SUCCESS:
            print("is_StopLiveVideo ERROR")
        else:
            self.capture_video = False
        return nRet

    def allocate_memory_buffer(self, buffers: int = 100):
        if len(self.MemInfo) > 0:
            self.free_memory()

        for x in range(buffers):
            self.MemInfo.append([ueye.c_mem_p(), ueye.c_int(0)])
            ret = ueye.is_AllocImageMem(
                self.hCam, c_int(self.width), c_int(self.height),
                self.bit_depth,
                self.MemInfo[x][0], self.MemInfo[x][1])
            ret = ueye.is_AddToSequence(
                self.hCam, self.MemInfo[x][0], self.MemInfo[x][1])

        ret = ueye.is_InitImageQueue(self.hCam, c_int(0))

    def unlock_buffer(self):
        return ueye.is_UnlockSeqBuf(
            self.hCam, self.current_id, self.current_buffer)

    def get_pitch(self):
        x = ueye.c_int()
        y = ueye.c_int()
        bits = ueye.c_int()

        pc_mem = ueye.c_mem_p()
        pid = ueye.c_int()
        ueye.is_GetActiveImageMem(self.hCam, pc_mem, pid)
        ueye.is_InquireImageMem(
            self.hCam, pc_mem, pid,
            x, y, bits, self.pitch)

        return self.pitch.value

    # Releases an image memory that was allocated using is_AllocImageMem()
    # and removes it from the driver management
    def free_memory(self):
        nRet = 0
        for x in range(len(self.MemInfo)):
            nRet += ueye.is_FreeImageMem(
                self.hCam, self.MemInfo[x][0], self.MemInfo[x][1])
        self.MemInfo.clear()
        self.memory_allocated = False
        return nRet

    def is_WaitForNextImage(self, wait=0, log=True):
        nret = ueye.is_WaitForNextImage(
            self.hCam, wait,
            self.current_buffer, self.current_id)
        if nret == ueye.IS_SUCCESS:
            if log:
                logging.debug("is_WaitForNextImage, IS_SUCCESS: {}"
                              .format(nret))
        elif nret == ueye.IS_TIMED_OUT:
            if log:
                logging.debug("is_WaitForNextImage, IS_TIMED_OUT: {}"
                              .format(nret))
        elif nret == ueye.IS_CAPTURE_STATUS:
            if log:
                logging.debug("is_WaitForNextImage, IS_CAPTURE_STATUS: {}"
                              .format(nret))
            self.CaptureStatusInfo = ueye.UEYE_CAPTURE_STATUS_INFO()
            nRet = ueye.is_CaptureStatus(
                self.hCam, ueye.IS_CAPTURE_STATUS_INFO_CMD_GET,
                self.CaptureStatusInfo, sizeof(self.CaptureStatusInfo))
            if nRet == ueye.IS_SUCCESS:
                ueye.is_CaptureStatus(
                    self.hCam, ueye.IS_CAPTURE_STATUS_INFO_CMD_RESET,
                    None, 0)
        return nret

    # Enables the queue mode for existing image memory sequences
    def enable_queue_mode(self):
        nRet = ueye.is_InquireImageMem(
            self.hCam, self.pcImageMemory,
            self.MemID, self.width,
            self.height, self.bit_depth, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

        return nRet

    def get_data(self):
        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        return ueye.get_data(
            self.current_buffer,
            self.width, self.height,
            self.bit_depth,
            self.pitch, copy=False)

    # Disables the hCam camera handle and releases the data structures
    # and memory areas taken up by the uEye camera
    def dispose(self):
        nRet = ueye.is_ExitCamera(self.hCam)
        return nRet
