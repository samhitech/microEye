from ctypes import c_uint
from pyueye import ueye
import numpy as np


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
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.set_rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        # 24: bits per pixel for color mode;
        # take 8 bits per pixel for monochrome
        self.nBitsPerPixel = ueye.INT(24)
        # 3: channels for color mode(RGB);
        # take 1 channel for monochrome
        self.channels = 3
        # Y8/RGB16/RGB24/REG32
        self.m_nColorMode = ueye.INT()
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
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
        cam_count = ueye.c_int(0)
        nRet = ueye.is_GetNumberOfCameras(cam_count)
        if nRet == ueye.IS_SUCCESS and cam_count > 0:
            pucl = ueye.UEYE_CAMERA_LIST(
                ueye.UEYE_CAMERA_INFO * cam_count.value)
            pucl.dwCount = cam_count.value
            if (ueye.is_GetCameraList(pucl) == ueye.IS_SUCCESS):
                cam_list = []
                for index in range(cam_count.value):
                    cam_list.append({
                        "camID": pucl.uci[index].dwCameraID.value,
                        "devID": pucl.uci[index].dwDeviceID.value,
                        "senID": pucl.uci[index].dwSensorID.value,
                        "Status": pucl.uci[index].dwStatus.value,
                        "InUse": pucl.uci[index].dwInUse.value,
                        "Model": pucl.uci[index].Model.decode('utf-8'),
                        "Serial": pucl.uci[index].SerNo.decode('utf-8')})
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
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == \
                ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(
                self.hCam,
                self.nBitsPerPixel,
                self.m_nColorMode)
            self.bytes_per_pixel = int(np.ceil(self.nBitsPerPixel / 8))
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == \
                ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(np.ceil(self.nBitsPerPixel / 8))
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == \
                ueye.IS_COLORMODE_MONOCHROME:
            # for mono camera models that uses CM_MONO12 mode
            self.m_nColorMode = ueye.IS_CM_MONO12
            self.nBitsPerPixel = ueye.INT(12)
            self.bytes_per_pixel = int(np.ceil(self.nBitsPerPixel / 8))
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(np.ceil(self.nBitsPerPixel / 8))
            print("IS_COLORMODE_MONOCHROME Else: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)

        nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

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
    # width and height and its color depth defined by nBitsPerPixel
    def allocate_memory(self):
        nRet = ueye.is_AllocImageMem(
            self.hCam,
            self.width,
            self.height,
            self.nBitsPerPixel,
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
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)
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

    # Enables the queue mode for existing image memory sequences
    def enable_queue_mode(self):
        nRet = ueye.is_InquireImageMem(
            self.hCam, self.pcImageMemory,
            self.MemID, self.width,
            self.height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

        return nRet

    def get_data(self):
        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        return ueye.get_data(
            self.pcImageMemory,
            self.width, self.height,
            self.nBitsPerPixel,
            self.pitch, copy=False)

    # Releases an image memory that was allocated using is_AllocImageMem()
    # and removes it from the driver management
    def free_memory(self):
        nRet = ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)
        self.memory_allocated = False
        return nRet

    # Disables the hCam camera handle and releases the data structures
    # and memory areas taken up by the uEye camera
    def dispose(self):
        nRet = ueye.is_ExitCamera(self.hCam)
        return nRet
