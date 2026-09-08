import contextlib
import logging
import threading
from ctypes import byref, c_double, c_int32, c_void_p, create_string_buffer, sizeof
from enum import Enum
from typing import Any, Optional

import numpy as np

import microEye.hardware.cams.dcam.dcamapi4 as dcamapi4
from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.micam import miCamera

logger = logging.getLogger(__name__)

# ==== DCAMAPI helper functions ====


def dcammisc_setupframe(hdcam, bufframe: dcamapi4.DCAMBUF_FRAME):
    '''Setup DCAMBUF_FRAME instance.

    Setup DCAMBUF_FRAME instance based on camera setting with hdcam.

    Args:
        hdcam (c_void_p): DCAM handle.
        bufframe (DCAMBUF_FRAME): Frame information.

    Returns:
        DCAMERR: DCAMERR value.
    '''
    fValue = dcamapi4.ct.c_double()
    idprop = dcamapi4.DCAM_IDPROP.IMAGE_PIXELTYPE
    err = dcamapi4.dcamprop_getvalue(hdcam, idprop, byref(fValue))
    if not err.is_failed():
        bufframe.type = int(fValue.value)

        idprop = dcamapi4.DCAM_IDPROP.IMAGE_WIDTH
        err = dcamapi4.dcamprop_getvalue(hdcam, idprop, byref(fValue))
        if not err.is_failed():
            bufframe.width = int(fValue.value)

            idprop = dcamapi4.DCAM_IDPROP.IMAGE_HEIGHT
            err = dcamapi4.dcamprop_getvalue(hdcam, idprop, byref(fValue))
            if not err.is_failed():
                bufframe.height = int(fValue.value)

                idprop = dcamapi4.DCAM_IDPROP.FRAMEBUNDLE_MODE
                err = dcamapi4.dcamprop_getvalue(hdcam, idprop, byref(fValue))
                if (
                    not err.is_failed()
                    and int(fValue.value) == dcamapi4.DCAMPROP.MODE.ON
                ):
                    idprop = dcamapi4.DCAM_IDPROP.FRAMEBUNDLE_ROWBYTES
                    err = dcamapi4.dcamprop_getvalue(hdcam, idprop, byref(fValue))
                    if not err.is_failed():
                        bufframe.rowbytes = int(fValue.value)
                else:
                    idprop = dcamapi4.DCAM_IDPROP.IMAGE_ROWBYTES
                    err = dcamapi4.dcamprop_getvalue(hdcam, idprop, byref(fValue))
                    if not err.is_failed():
                        bufframe.rowbytes = int(fValue.value)

    return err


def dcammisc_alloc_ndarray(frame: dcamapi4.DCAMBUF_FRAME, framebundlenum=1, viewnum=1):
    '''Allocate NumPy ndarray.

    Allocate NumPy ndarray based on information of DCAMBUF_FRAME.

    Args:
        frame (DCAMBUF_FRAME): Frame information.
        framebundlenum (int): Frame Bundle number.
        viewnum(int) : Number of views.

    Returns:
        NumPy ndarray: NumPy ndarray buffer.
        bool: False if failed to allocate NumPy ndarray buffer.
    '''
    height = frame.height * framebundlenum * viewnum

    if frame.type == dcamapi4.DCAM_PIXELTYPE.MONO16:
        return np.zeros((height, frame.width), dtype='uint16')

    if frame.type == dcamapi4.DCAM_PIXELTYPE.MONO8:
        return np.zeros((height, frame.width), dtype='uint8')

    return False


# ==== declare Dcamapi class ====


class Dcamapi:
    # class instance
    __lasterr = (
        dcamapi4.DCAMERR.SUCCESS
    )  # the last error from functions with dcamapi_ prefix.
    __bInitialized = (
        False  # Once Dcamapi.init() is called, then True.  Dcamapi.uninit() reset this.
    )
    __devicecount = 0

    @classmethod
    def __result(cls, errvalue):
        '''Keep last error code.

        Internal use. Keep last error code.
        '''
        if errvalue < 0:
            cls.__lasterr = errvalue
            return False

        return True

    @classmethod
    def lasterr(cls):
        '''
        Return last error code of Dcamapi member functions.
        '''
        if isinstance(cls.__lasterr, dcamapi4.DCAMERR):
            return cls.__lasterr
        else:
            return dcamapi4.DCAMERR(cls.__lasterr)

    @classmethod
    def init(cls, *initparams):
        '''
        Initialize DCAM-API.

        Do not call this when Dcam object
        exists because constructor of Dcam ececute this.

        After calling close(), call this again if you need to resume measurement.

        Returns
        -------
        bool:
            True if initialize DCAM-API was succeeded.

            False if dcamapi_init() returned DCAMERR except SUCCESS.

            lasterr() returns the DCAMERR value.
        '''
        if not dcamapi4.LIBRARY_LOADED:
            return False

        if cls.__bInitialized:
            return cls.__result(
                dcamapi4.DCAMERR.ALREADYINITIALIZED
            )  # dcamapi_init() is called. New Error.


        paraminit = dcamapi4.DCAMAPI_INIT()
        err = dcamapi4.dcamapi_init(byref(paraminit))
        cls.__bInitialized = True
        if cls.__result(err) is False:
            return False

        cls.__devicecount = paraminit.iDeviceCount
        return True

    @classmethod
    def uninit(cls):
        '''
        Uninitialize DCAM-API.

        After using DCAM-API, call this function to close all resources.

        Returns
        -------
        bool:
            True if uninitialize DCAM-API was succeeded.
        '''
        if cls.__bInitialized:
            dcamapi4.dcamapi_uninit()
            cls.__lasterr = dcamapi4.DCAMERR.SUCCESS
            cls.__bInitialized = False
            cls.__devicecount = 0

        return True

    @classmethod
    def get_devicecount(cls):
        '''
        Return number of connected cameras.

        Returns
        -------
        int | bool :
            - Number of connected cameras.
            - False if not initialized.
        '''
        if not cls.__bInitialized:
            return False

        return cls.__devicecount


class _DcamApiManager:
    '''
    Reference-counted wrapper around Dcamapi.init()/uninit().

    Dcamapi.init()/uninit() operate on the whole DCAM-API session, not on a
    single camera, and DCAM does not like being re-initialised while a
    Dcam handle is still open.

    Every dcamera instance (and the class-method device enumerator) calls
    acquire()/release() instead of touching Dcamapi directly, so the API
    is initialised exactly once no matter how many cameras/panels are open,
    and is only torn down when the last one closes.
    '''

    _lock = threading.Lock()
    _refcount = 0

    @classmethod
    def acquire(cls) -> bool:
        with cls._lock:
            if cls._refcount == 0:
                if not Dcamapi.init():
                    if Dcamapi.lasterr() == dcamapi4.DCAMERR.NOCAMERA:
                        logger.warning('DCAM-API: no camera found.')
                    else:
                        Dcamapi.lasterr()._logger_error(
                            logger, 'Dcamapi.init() failed with error'
                        )
                    Dcamapi.uninit()
                    return False
            cls._refcount += 1
            return True

    @classmethod
    def release(cls):
        with cls._lock:
            if cls._refcount <= 0:
                return
            cls._refcount -= 1
            if cls._refcount == 0:
                Dcamapi.uninit()

    @classmethod
    def device_count(cls) -> int:
        with cls._lock:
            if cls._refcount == 0:
                # temporary init just to enumerate
                if not Dcamapi.init():
                    return 0
                n = Dcamapi.get_devicecount() or 0
                Dcamapi.uninit()
                return n
            return Dcamapi.get_devicecount() or 0


# ==== Dcam class ====


class Dcam:
    def __init__(self, iDevice=0):
        self.__lasterr = dcamapi4.DCAMERR.SUCCESS
        self.__iDevice = iDevice
        self.__hdcam = 0
        self.__hdcamwait = 0
        self.__bufframe = dcamapi4.DCAMBUF_FRAME()

    def __repr__(self):
        return 'Dcam()'

    def __result(self, errvalue):
        '''
        Keep last error code.

        Internal use. Keep last error code.
        '''
        if errvalue < 0:
            self.__lasterr = errvalue
            return False

        return True

    def lasterr(self):
        '''
        Return last error code of Dcam member functions.
        '''
        if isinstance(self.__lasterr, dcamapi4.DCAMERR):
            return self.__lasterr
        else:
            return dcamapi4.DCAMERR(self.__lasterr)

    def is_opened(self):
        '''
        Check DCAM handle is opened.

        Returns:
            bool: True if DCAM handle is opened. False if DCAM handle is not opened.
        '''
        return self.__hdcam != 0

    def dev_open(self, index=-1):
        '''
        Get DCAM handle for controling camera.

        After calling close(), call this again if you need to resume measurement.

        Args:
            index (int): Device index.

        Returns:
            bool: True if get DCAM handle was succeeded.
            False if dcamdev_open() returned DCAMERR except SUCCESS.
            lasterr() returns the DCAMERR value.
        '''
        if self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.ALREADYOPENED
            )  # instance is already opened. New Error.

        paramopen = dcamapi4.DCAMDEV_OPEN()
        if index >= 0:
            paramopen.index = index
        else:
            paramopen.index = self.__iDevice

        ret = self.__result(dcamapi4.dcamdev_open(byref(paramopen)))
        if ret is False:
            return False

        self.__hdcam = paramopen.hdcam
        return True

    def dev_close(self):
        '''
        Close DCAM handle.
        Call this if you need to close the current device.

        Returns:
            bool: True if close DCAM handle was succeeded.
        '''
        if self.is_opened():
            self.__close_hdcamwait()
            dcamapi4.dcamdev_close(self.__hdcam)
            self.__lasterr = dcamapi4.DCAMERR.SUCCESS
            self.__hdcam = 0

        return True

    def dev_getstring(self, idstr: dcamapi4.DCAM_IDSTR):
        '''
        Get string of device.

        Args:
            idstr (DCAM_IDSTR): String id.

        Returns:
            string: Device information specified by DCAM_IDSTR.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        hdcam = self.__hdcam if self.is_opened() else self.__iDevice

        paramdevstr = dcamapi4.DCAMDEV_STRING()
        paramdevstr.iString = idstr
        paramdevstr.alloctext(256)

        ret = self.__result(dcamapi4.dcamdev_getstring(hdcam, byref(paramdevstr)))
        if ret is False:
            return False

        return paramdevstr.text.decode()

    def dev_getcapability(self):
        '''
        Get capability of function

        Returns:
            DCAMDEV_CAPABILITY: Capability of function
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        capability = dcamapi4.DCAMDEV_CAPABILITY()
        ret = self.__result(
            dcamapi4.dcamdev_getcapability(self.__hdcam, byref(capability))
        )
        if ret is False:
            return False

        return capability

    # dcamprop functions

    def prop_getattr(self, idprop: dcamapi4.DCAM_IDPROP):
        '''
        Get property attribute.

        args:
            idprop (DCAM_IDPROP): Property id.

        Returns:
            DCAMPROP_ATTR: Attribute information of the property.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        propattr = dcamapi4.DCAMPROP_ATTR()
        propattr.iProp = idprop
        ret = self.__result(dcamapi4.dcamprop_getattr(self.__hdcam, byref(propattr)))
        if ret is False:
            return False

        return propattr

    def prop_getvalue(self, idprop: dcamapi4.DCAM_IDPROP):
        '''
        Get property value.

        args:
            idprop (DCAM_IDPROP): Property id.

        Returns:
            float: Property value of property id.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        cDouble = c_double()
        ret = self.__result(
            dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, byref(cDouble))
        )
        if ret is False:
            return False

        return cDouble.value

    def prop_setvalue(self, idprop: dcamapi4.DCAM_IDPROP, fValue):
        '''
        Set property value.

        args:
            idprop (DCAM_IDPROP): Property id.
            fValue (float): Setting value.

        Returns:
            bool: True if set property value was succeeded.
            False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        ret = self.__result(dcamapi4.dcamprop_setvalue(self.__hdcam, idprop, fValue))
        return ret is not False

    def prop_setgetvalue(self, idprop: dcamapi4.DCAM_IDPROP, fValue, option=0):
        '''
        Set and get property value.

        args:
            idprop (DCAM_IDPROP): Property id.
            fValue (float): Input value for setting and receive actual set value by ref.

        Returns:
            float: Accurate value set in device.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        cDouble = c_double(fValue)
        cOption = c_int32(option)
        ret = self.__result(
            dcamapi4.dcamprop_setgetvalue(self.__hdcam, idprop, byref(cDouble), cOption)
        )
        if ret is False:
            return False

        return cDouble.value

    def prop_queryvalue(self, idprop: dcamapi4.DCAM_IDPROP, fValue, option=0):
        '''
        Query property value.

        Args:
            idprop (DCAM_IDPROP): Property id.
            fValue (float): Value of property.

        Returns:
            float: Property value specified by option.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        cDouble = c_double(fValue)
        cOption = c_int32(option)
        ret = self.__result(
            dcamapi4.dcamprop_queryvalue(self.__hdcam, idprop, byref(cDouble), cOption)
        )
        if ret is False:
            return False

        return cDouble.value

    def prop_getnextid(self, idprop: dcamapi4.DCAM_IDPROP):
        '''
        Get next property id.

        Args:
            idprop (DCAM_IDPROP): Property id.

        Returns:
            DCAM_IDPROP: Next property id.
            bool: False if no more property or error happened.
            lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        cIdprop = c_int32(idprop)
        cOption = c_int32(0)  # search next ID

        ret = self.__result(
            dcamapi4.dcamprop_getnextid(self.__hdcam, byref(cIdprop), cOption)
        )
        if ret is False:
            return False

        return cIdprop.value

    def prop_getname(self, idprop: dcamapi4.DCAM_IDPROP):
        '''
        Get name of property.

        Args:
            idprop (DCAM_IDPROP): Property id.

        Returns:
            string: Caracter string of property id.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        textbuf = create_string_buffer(256)
        ret = self.__result(
            dcamapi4.dcamprop_getname(self.__hdcam, idprop, textbuf, sizeof(textbuf))
        )
        if ret is False:
            return False

        return textbuf.value.decode()

    def prop_getvaluetext(self, idprop: dcamapi4.DCAM_IDPROP, fValue):
        '''
        Get text of property value.

        Args:
            idprop (DCAM_IDPROP): Property id.
            fValue (float): Setting value.

        Returns:
            string: Caracter string of property value.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        paramvaluetext = dcamapi4.DCAMPROP_VALUETEXT()
        paramvaluetext.iProp = idprop
        paramvaluetext.value = fValue
        paramvaluetext.alloctext(256)

        ret = self.__result(
            dcamapi4.dcamprop_getvaluetext(self.__hdcam, byref(paramvaluetext))
        )
        if ret is False:
            return False

        return paramvaluetext.text.decode()

    # dcambuf functions

    def buf_alloc(self, nFrame):
        '''
        Alloc DCAM internal buffer.

        Arg:
            nFrame (int): Number of frames.

        Returns:
            bool: True if buffer is prepared.
            False if buffer is not prepared.
            lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        cFrame = c_int32(nFrame)
        ret = self.__result(dcamapi4.dcambuf_alloc(self.__hdcam, cFrame))
        if ret is False:
            return False

        return self.__result(dcammisc_setupframe(self.__hdcam, self.__bufframe))

    def buf_release(self):
        '''
        Release DCAM internal buffer.

        Returns:
            bool: True if release DCAM internal buffser was succeeded.
            False if error happens during releasing buffer.
            lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        cOption = c_int32(0)
        return self.__result(dcamapi4.dcambuf_release(self.__hdcam, cOption))

    def buf_getframe(self, iFrame):
        '''
        Return DCAMBUF_FRAME instance with image data specified by iFrame.

        Arg:
            iFrame (int): Index of target frame.

        Returns:
            (aFrame, npBuf): aFrame is DCAMBUF_FRAME, npBuf is NumPy buffer.
            bool: False if error happens. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        framebundlenum = 1

        fValue = c_double()
        err = dcamapi4.dcamprop_getvalue(
            self.__hdcam, dcamapi4.DCAM_IDPROP.FRAMEBUNDLE_MODE, byref(fValue)
        )
        if not err.is_failed() and int(fValue.value) == dcamapi4.DCAMPROP.MODE.ON:
            err = dcamapi4.dcamprop_getvalue(
                self.__hdcam, dcamapi4.DCAM_IDPROP.FRAMEBUNDLE_NUMBER, byref(fValue)
            )
            if not err.is_failed():
                framebundlenum = int(fValue.value)
            else:
                return False

        viewnum = 1

        err = dcamapi4.dcamprop_getvalue(
            self.__hdcam, dcamapi4.DCAM_IDPROP.NUMBEROF_VIEW, byref(fValue)
        )
        if not err.is_failed():
            viewnum = int(fValue.value)

        npBuf = dcammisc_alloc_ndarray(self.__bufframe, framebundlenum, viewnum)

        if npBuf is False:
            return self.__result(dcamapi4.DCAMERR.INVALIDPIXELTYPE)

        aFrame = dcamapi4.DCAMBUF_FRAME()
        aFrame.iFrame = iFrame

        aFrame.buf = npBuf.ctypes.data_as(c_void_p)
        aFrame.rowbytes = self.__bufframe.rowbytes
        aFrame.type = self.__bufframe.type
        aFrame.width = self.__bufframe.width
        aFrame.height = self.__bufframe.height

        ret = self.__result(dcamapi4.dcambuf_copyframe(self.__hdcam, byref(aFrame)))
        if ret is False:
            return False

        return (aFrame, npBuf)

    def buf_getframedata(self, iFrame):
        '''
        Return NumPy buffer of image data specified by iFrame.

        Arg:
            iFrame (int): Index of target frame.

        Returns:
            npBuf: NumPy buffer.
            bool: False if error happens. lasterr() returns the DCAMERR value.
        '''
        ret = self.buf_getframe(iFrame)
        if ret is False:
            return False

        return ret[1]

    def buf_getlastframedata(self):
        '''
        Return NumPy buffer of image data of last updated frame.

        Returns:
            npBuf: NumPy buffer.
            bool: False if error happens. lasterr() returns the DCAMERR value.
        '''
        return self.buf_getframedata(-1)

    # dcamcap functions

    def cap_start(self, bSequence=True):
        '''
        Start capturing.

        Arg:
            bSequence (bool): False means SNAPSHOT, others means SEQUENCE.

        Returns:
            bool: True if start capture. False if error happened.
            lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        if bSequence:
            mode = dcamapi4.DCAMCAP_START.SEQUENCE
        else:
            mode = dcamapi4.DCAMCAP_START.SNAP

        return self.__result(dcamapi4.dcamcap_start(self.__hdcam, mode))

    def cap_snapshot(self):
        '''
        Capture snapshot. Get the frames specified in buf_alloc().

        Returns:
            bool: True if start snapshot. False if error happened.
            lasterr() returns the DCAMERR value.
        '''
        return self.cap_start(False)

    def cap_stop(self):
        '''
        Stop capturing.

        Returns:
            bool: True if Stop capture. False if error happened.
            lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        return self.__result(dcamapi4.dcamcap_stop(self.__hdcam))

    def cap_status(self):
        '''
        Get capture status.

        Returns:
            DCAMCAP_STATUS: Current capturing status.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        cStatus = c_int32()
        ret = self.__result(dcamapi4.dcamcap_status(self.__hdcam, byref(cStatus)))
        if ret is False:
            return False

        return cStatus.value

    def cap_transferinfo(self):
        '''
        Get transfer info.

        Returns:
            DCAMCAP_TRANSFERINFO: Current image transfer status.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        paramtransferinfo = dcamapi4.DCAMCAP_TRANSFERINFO()
        ret = self.__result(
            dcamapi4.dcamcap_transferinfo(self.__hdcam, byref(paramtransferinfo))
        )
        if ret is False:
            return False

        return paramtransferinfo

    def cap_firetrigger(self):
        '''
        Fire software trigger.

        Returns:
            bool: True if firing trigger was succeeded.
            False if error happened. lasterr() returns the DCAMERR value.
        '''
        if not self.is_opened():
            return self.__result(
                dcamapi4.DCAMERR.INVALIDHANDLE
            )  # instance is not opened yet.

        cOption = c_int32(0)
        ret = self.__result(dcamapi4.dcamcap_firetrigger(self.__hdcam, cOption))
        return ret is not False

    # dcamwait functions

    def __open_hdcamwait(self):
        '''
        Get DCAMWAIT handle.

        Returns:
            bool: True if get DCAMWAIT handle was succeeded.
            False if error happened. lasterr() returns the DCAMERR value.
        '''
        if self.__hdcamwait != 0:
            return True

        paramwaitopen = dcamapi4.DCAMWAIT_OPEN()
        paramwaitopen.hdcam = self.__hdcam
        ret = self.__result(dcamapi4.dcamwait_open(byref(paramwaitopen)))
        if ret is False:
            return False

        if paramwaitopen.hwait == 0:
            return self.__result(dcamapi4.DCAMERR.INVALIDWAITHANDLE)

        self.__hdcamwait = paramwaitopen.hwait
        return True

    def __close_hdcamwait(self):
        '''
        Close DCAMWAIT handle.

        Returns:
            bool: True if close DCAMWAIT handle was succeeded.
            False if error happened. lasterr() returns the DCAMERR value.
        '''

        if self.__hdcamwait == 0:
            return True

        ret = self.__result(dcamapi4.dcamwait_close(self.__hdcamwait))
        if ret is False:
            return False

        self.__hdcamwait = 0
        return True

    def wait_event(self, eventmask: dcamapi4.DCAMWAIT_CAPEVENT, timeout_millisec):
        '''
        Wait specified event.

        Arg:
            eventmask (DCAMWAIT_CAPEVENT): Event mask to wait.
            timeout_millisec (int): Timeout by milliseconds.

        Returns:
            DCAMWAIT_CAPEVENT: Happened event.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        '''
        ret = self.__open_hdcamwait()
        if ret is False:
            return False

        paramwaitstart = dcamapi4.DCAMWAIT_START()
        paramwaitstart.eventmask = eventmask
        paramwaitstart.timeout = timeout_millisec
        ret = self.__result(
            dcamapi4.dcamwait_start(self.__hdcamwait, byref(paramwaitstart))
        )
        if ret is False:
            return False

        return paramwaitstart.eventhappened

    def wait_capevent_frameready(self, timeout_millisec):
        '''
        Wait DCAMWAIT_CAPEVENT.FRAMEREADY event.

        Arg:
            timeout_millisec (int): Timeout by milliseconds.

        Returns:
            bool: True if wait capture.
            False if error happened. lasterr() returns the DCAMERR value.
        '''
        ret = self.wait_event(dcamapi4.DCAMWAIT_CAPEVENT.FRAMEREADY, timeout_millisec)

        return ret is not False


class DcamParams(Enum):
    FREERUN = 'Acquisition.Freerun'
    STOP = 'Acquisition.Stop'
    BUFFER_FRAMES = 'Acquisition.Ring Buffer Frames'

    FRAMERATE = 'Acquisition Settings.Frame Rate'

    BITS_PER_CHANNEL = 'Acquisition Settings.Bits Per Channel'

    SENSOR_MODE = 'Acquisition Settings.Sensor Mode'
    TRIGGER_SOURCE = 'Acquisition Settings.Trigger Source'
    TRIGGER_MODE = 'Acquisition Settings.Trigger Mode'
    TRIGGER_ACTIVE = 'Acquisition Settings.Trigger Active'
    READOUT_SPEED = 'Acquisition Settings.Readout Speed'
    SUBARRAY_MODE = 'Acquisition Settings.Subarray Mode'
    BINNING = 'Acquisition Settings.Binning'
    SENSOR_COOLER = 'Acquisition Settings.Cooler'
    COOLER_FAN = 'Acquisition Settings.Cooler Fan'
    TEMPERATURE_TARGET = 'Acquisition Settings.Temperature Target'

    DEFECTCORRECT_MODE = 'Acquisition Settings.Defect Correct Mode'
    HOTPIXELCORRECT_LEVEL = 'Acquisition Settings.Hot Pixel Correct Level'
    DEFECTCORRECT_METHOD = 'Acquisition Settings.Defect Correct Method'

    CONVERSIONFACTOR_COEFF = 'Acquisition Settings.Conversion Factor Coefficient'
    SHUTTER_MODE = 'Acquisition Settings.Shutter Mode'

    GPIOCONNECTORS = 'GPIOs.Output Trigger Connectors'
    GPIO_CHANNELSYNC = 'GPIOs.Output Trigger Channel Sync'

    GPIO_1_ACTIVE = 'GPIOs.GPIO 1 Active'
    GPIO_1_SOURCE = 'GPIOs.GPIO 1 Source'
    GPIO_1_POLARITY = 'GPIOs.GPIO 1 Polarity'
    GPIO_1_KIND = 'GPIOs.GPIO 1 Kind'
    GPIO_1_PROGRAMABLESTART = 'GPIOs.GPIO 1 Programable Start'
    GPIO_1_BASESENSOR = 'GPIOs.GPIO 1 Base Sensor'
    GPIO_1_DELAY = 'GPIOs.GPIO 1 Delay'
    GPIO_1_PERIOD = 'GPIOs.GPIO 1 Period'

    GPIO_2_ACTIVE = 'GPIOs.GPIO 2 Active'
    GPIO_2_SOURCE = 'GPIOs.GPIO 2 Source'
    GPIO_2_POLARITY = 'GPIOs.GPIO 2 Polarity'
    GPIO_2_KIND = 'GPIOs.GPIO 2 Kind'
    GPIO_2_PROGRAMABLESTART = 'GPIOs.GPIO 2 Programable Start'
    GPIO_2_DELAY = 'GPIOs.GPIO 2 Delay'
    GPIO_2_PERIOD = 'GPIOs.GPIO 2 Period'

    GPIO_3_ACTIVE = 'GPIOs.GPIO 3 Active'
    GPIO_3_SOURCE = 'GPIOs.GPIO 3 Source'
    GPIO_3_POLARITY = 'GPIOs.GPIO 3 Polarity'
    GPIO_3_KIND = 'GPIOs.GPIO 3 Kind'
    GPIO_3_PROGRAMABLESTART = 'GPIOs.GPIO 3 Programable Start'
    GPIO_3_DELAY = 'GPIOs.GPIO 3 Delay'
    GPIO_3_PERIOD = 'GPIOs.GPIO 3 Period'

    PIXEL_WIDTH_UM = 'Stats.Pixel Width um'
    PIXEL_HEIGHT_UM = 'Stats.Pixel Height um'

    def __str__(self):
        return self.value.split('.')[-1]

    def get_path(self):
        return self.value.split('.')

    def get_parent(self):
        return '.'.join(self.value.split('.')[:-1])


class dcamera(miCamera):
    '''Adapter exposing a Hamamatsu DCAM camera through the miCamera API.'''

    def __init__(self, iDevice=0) -> None:
        super().__init__(iDevice)

        iDevice = int(iDevice)

        # Guards all non-reentrant access to the DCAM handle: property
        # get/set from the GUI thread vs. the capture loop running in the
        # acquisition worker thread.
        self._lock = threading.RLock()

        if not _DcamApiManager.acquire():
            raise RuntimeError('Failed to initialize DCAM-API.')

        self.dcam = Dcam(iDevice)
        if not self.dcam.dev_open():
            _DcamApiManager.release()
            self.dcam.lasterr()._logger_error(logger, 'Dcam.dev_open() failed')
            raise RuntimeError(
                f'Failed to open DCAM device #{iDevice}: {self.dcam.lasterr()}'
            )

        self.bytes_per_pixel = 2
        self.buffer_frames = 16  # ring-buffer depth for free-run capture

        self.exposure_range = [0.000017632, 10.000004662]
        self.exposure_increment = 0.0001
        self.exposure_unit = 's'

        # Overrun accounting (see wait_and_pop_frames). Reset on every
        # start_freerun() call so counts don't leak across runs.
        self._last_frame_count = -1
        self.dropped_frames = 0
        self._capturing = False

        self.initialize()

    def __enter__(self):
        self._lock.acquire()
        return self.dcam

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False

    def initialize(self):
        with self:
            self.name = self.dcam.dev_getstring(dcamapi4.DCAM_IDSTR.MODEL) or 'DCAM'
            self.serial = self.dcam.dev_getstring(dcamapi4.DCAM_IDSTR.CAMERAID) or 'NA'

            # Make sure subarray covers the full sensor before reading
            # width/height, then cache exposure + ROI state.
            with contextlib.suppress(Exception):
                self.dcam.prop_setvalue(
                    dcamapi4.DCAM_IDPROP.SUBARRAYMODE, dcamapi4.DCAMPROP.MODE.OFF
                )

            self._width = int(
                self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.IMAGE_WIDTH) or 512
            )
            self._height = int(
                self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.IMAGE_HEIGHT) or 512
            )

            self.exposure_unit = 's'

            self.get_exposure()
            self.get_framerate()
            self.get_sensor_dimensions()

        self.print_status()

    def close(self):
        '''Stop any running capture and release the DCAM handle + API ref.'''
        with contextlib.suppress(Exception):
            self.stop_freerun()

        with contextlib.suppress(Exception), self:
            self.dcam.dev_close()

        _DcamApiManager.release()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    @classmethod
    def get_camera_list(cls) -> list[dict[str, Any]]:
        cam_list = []
        if not _DcamApiManager.acquire():
            return cam_list
        try:
            n = Dcamapi.get_devicecount() or 0
            for i in range(n):
                dcam = Dcam(i)
                model = dcam.dev_getstring(dcamapi4.DCAM_IDSTR.MODEL) or 'NA'
                camid = dcam.dev_getstring(dcamapi4.DCAM_IDSTR.CAMERAID) or 'NA'
                cam_list.append(
                    {
                        'Camera ID': camid,
                        'Device ID': i,
                        'Sensor ID': 'NA',
                        'Status': 'NA',
                        'InUse': False,
                        'Model': model,
                        'Serial': camid,
                        'Driver': 'DCAM-API',
                        'FullName': f'{model} ({camid})',
                    }
                )
        finally:
            _DcamApiManager.release()
        return cam_list

    def get_metadata(self) -> dict[str, Any]:
        return {
            'CHANNEL_NAME': self.name,
            'DET_MANUFACTURER': 'Hamamatsu',
            'DET_MODEL': self.name,
            'DET_SERIAL': getattr(self, 'serial', 'NA'),
            'DET_TYPE': 'CMOS',
        }

    def get_exposure(self) -> float:
        with self:
            attr = self.dcam.prop_getattr(dcamapi4.DCAM_IDPROP.EXPOSURETIME)
            if attr is not False:
                self.exposure_range = [attr.valuemin, attr.valuemax]
                self.exposure_increment = attr.valuestep or 1e-6
            else:
                self.exposure_range = [1e-6, 10.0]
                self.exposure_increment = 1e-6

            value = self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.EXPOSURETIME)
            if value is not False:
                self.exposure_current = value
        return self.exposure_current

    def set_exposure(self, seconds: float) -> float:
        with self:
            value = self.dcam.prop_setgetvalue(
                dcamapi4.DCAM_IDPROP.EXPOSURETIME, float(seconds)
            )
            if value is not False:
                self.exposure_current = value
            else:
                self.dcam.lasterr()._logger_error(logger, 'set exposure failed')
        return self.exposure_current

    def get_framerate(self) -> float:
        with self:
            attr = self.dcam.prop_getattr(dcamapi4.DCAM_IDPROP.INTERNALFRAMERATE)
            if attr is not False:
                self.framerate_range = [attr.valuemin, attr.valuemax]
                self.framerate_increment = attr.valuestep or 1e-6
            else:
                self.framerate_range = [0.0, 1000.0]
                self.framerate_increment = 1e-6

            value = self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.INTERNALFRAMERATE)
            if value is not False and value > 0:
                self.framerate_current = value
        return self.framerate_current

    def set_framerate(self, fps: float) -> float:
        with self:
            value = self.dcam.prop_setgetvalue(
                dcamapi4.DCAM_IDPROP.INTERNALFRAMERATE, float(fps)
            )
            if value is not False:
                self.framerate_current = value
            else:
                self.dcam.lasterr()._logger_error(logger, 'set framerate failed')
        return self.framerate_current

    def get_temperature(self) -> float:
        with self:
            value = self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.SENSORTEMPERATURE)
        self.temperature = value if value is not False else -127
        return self.temperature

    def _align(self, value, vmin, step):
        if not step:
            return value
        return vmin + round((value - vmin) / step) * step

    def set_roi(self, x: int, y: int, width: int, height: int):
        with self:
            # SUBARRAY H/V POS/SIZE can only be changed while
            # SUBARRAYMODE == OFF; it is then switched back ON to apply.
            self.dcam.prop_setvalue(
                dcamapi4.DCAM_IDPROP.SUBARRAYMODE, dcamapi4.DCAMPROP.MODE.OFF
            )

            for idprop, value in (
                (dcamapi4.DCAM_IDPROP.SUBARRAYHPOS, x),
                (dcamapi4.DCAM_IDPROP.SUBARRAYVPOS, y),
                (dcamapi4.DCAM_IDPROP.SUBARRAYHSIZE, width),
                (dcamapi4.DCAM_IDPROP.SUBARRAYVSIZE, height),
            ):
                attr = self.dcam.prop_getattr(idprop)
                aligned = (
                    self._align(value, attr.valuemin, attr.valuestep)
                    if attr is not False
                    else value
                )
                self.dcam.prop_setvalue(idprop, aligned)

            self.dcam.prop_setvalue(
                dcamapi4.DCAM_IDPROP.SUBARRAYMODE, dcamapi4.DCAMPROP.MODE.ON
            )

            self._width = int(self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.IMAGE_WIDTH))
            self._height = int(
                self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.IMAGE_HEIGHT)
            )

    def get_roi(self) -> tuple[int, int, int, int]:
        with self:
            x = int(self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.SUBARRAYHPOS) or 0)
            y = int(self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.SUBARRAYVPOS) or 0)

            self._width = int(self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.IMAGE_WIDTH))
            self._height = int(
                self.dcam.prop_getvalue(dcamapi4.DCAM_IDPROP.IMAGE_HEIGHT)
            )
        return (x, y, self._width, self._height)

    def get_sensor_dimensions(self):
        with self:
            self.max_dim = (
                int(
                    self.dcam.prop_getvalue(
                        dcamapi4.DCAM_IDPROP.IMAGEDETECTOR_PIXELNUMHORZ
                    )
                ),
                int(
                    self.dcam.prop_getvalue(
                        dcamapi4.DCAM_IDPROP.IMAGEDETECTOR_PIXELNUMVERT
                    )
                ),
            )
            attr_w = self.dcam.prop_getattr(dcamapi4.DCAM_IDPROP.SUBARRAYHSIZE)
            attr_x = self.dcam.prop_getattr(dcamapi4.DCAM_IDPROP.SUBARRAYHPOS)
            attr_h = self.dcam.prop_getattr(dcamapi4.DCAM_IDPROP.SUBARRAYVSIZE)
            attr_y = self.dcam.prop_getattr(dcamapi4.DCAM_IDPROP.SUBARRAYVPOS)

            self.subarrayh_range = (
                int(attr_w.valuemin) if attr_w is not False else 4,
                int(attr_w.valuemax) if attr_w is not False else 2304,
            )
            self.subarrayv_range = (
                int(attr_h.valuemin) if attr_h is not False else 4,
                int(attr_h.valuemax) if attr_h is not False else 2304,
            )
            self.subarrayh_pos_range = (
                int(attr_x.valuemin) if attr_x is not False else 0,
                int(attr_x.valuemax) if attr_x is not False else 2300,
            )
            self.subarrayv_pos_range = (
                int(attr_y.valuemin) if attr_y is not False else 0,
                int(attr_y.valuemax) if attr_y is not False else 2300,
            )

    def get_sensor_pixel_width_um(self) -> float:
        with self:
            value = self.dcam.prop_getvalue(
                dcamapi4.DCAM_IDPROP.IMAGEDETECTOR_PIXELWIDTH
            )
        return value if value is not False else 6.5

    def get_sensor_pixel_height_um(self) -> float:
        with self:
            value = self.dcam.prop_getvalue(
                dcamapi4.DCAM_IDPROP.IMAGEDETECTOR_PIXELHEIGHT
            )
        return value if value is not False else 6.5

    def reset_roi(self):
        self.set_roi(0, 0, self.max_dim[0], self.max_dim[1])

        with self:
            self.dcam.prop_setvalue(
                dcamapi4.DCAM_IDPROP.SUBARRAYMODE, dcamapi4.DCAMPROP.MODE.OFF
            )

    def suggest_buffer_frames(
        self,
        expected_stall_ms: float = 250.0,
        min_frames: int = 16,
        max_frames: int = 12,
    ) -> int:
        """
        Suggest a ring-buffer depth large enough to absorb a consumer
        stall of `expected_stall_ms` at the current exposure time, so a
        transient display/save hiccup of roughly that duration does not
        cause an overrun.

        Parameters
        ----------
        expected_stall_ms : float
            Worst-case pause you want to tolerate without dropping a
            frame (GUI repaint, a save-thread fsync, etc).
        min_frames, max_frames : int
            Sane floor/ceiling so a very short exposure doesn't compute
            an unreasonably large allocation, and a very long exposure
            doesn't compute a useless 1-frame buffer.
        """
        frame_period_s = max(self.exposure_current, 1e-6)
        frames_for_stall = int(round((expected_stall_ms / 1000.0) / frame_period_s))
        # make frames_for_stall a power of two for better
        # memory alignment and to avoid odd buffer sizes
        frames_for_stall = 2 ** int(np.ceil(np.log2(frames_for_stall)))
        return max(min_frames, min(max_frames, frames_for_stall))

    def start_freerun(self, n_buffers: Optional[int] = None):
        with self:
            self.dcam.buf_release()
            n = n_buffers or self.buffer_frames
            if not self.dcam.buf_alloc(n):
                self.dcam.lasterr()._logger_error(logger, 'buf_alloc() failed')
                return False

            if not self.dcam.cap_start(bSequence=True):
                self.dcam.lasterr()._logger_error(logger, 'cap_start() failed')
                self.dcam.buf_release()
                return False

            self.buffer_frames = n
            self._last_frame_count = -1
            self.dropped_frames = 0
            self._capturing = True
            self.acquisition = True
        return True

    def stop_freerun(self):
        with self:
            if self._capturing:
                self.dcam.cap_stop()
                self.dcam.buf_release()
                self._capturing = False
            self.acquisition = False

    def wait_and_pop_frames(self, timeout_ms=1000) -> list[np.ndarray]:
        '''
        Block until at least one frame is ready, then return every new
        frame collected since the previous call, oldest first, without
        holding the camera lock for the duration of the wait.

        Returns an empty list on timeout so the caller can keep polling
        (e.g. to check a stop flag) without treating it as an error.

        Overrun handling: cap_transferinfo().nFrameCount is a cumulative,
        monotonically increasing counter of every frame DCAM has captured
        since cap_start() - unlike the ring index, its delta between two
        polls is an unambiguous count of frames produced in that interval,
        even across multiple full buffer wraps. That delta decides how
        many frames to read and whether any were unavoidably lost (delta
        > buffer_frames means the ring wrapped past frames that were
        never read out - those are gone, no way to recover them after the
        fact). Every overrun is counted in self.dropped_frames and
        logged - it is never silently absorbed.
        '''
        if not self._capturing:
            return []

        # wait_capevent_frameready() itself is safe to call unlocked: it
        # only touches the DCAMWAIT handle, not properties, so it does not
        # need to serialize against GUI-thread property access, and NOT
        # holding the lock here is what lets exposure/ROI/etc. still be
        # queried while a long exposure is in flight.
        ready = self.dcam.wait_capevent_frameready(timeout_ms)
        if not ready:
            err = self.dcam.lasterr()
            if not err.is_timeout():
                err._logger_error(logger, 'wait_capevent_frameready() failed')
                self._capturing = False
            return []

        frames: list[np.ndarray] = []
        with self:
            info = self.dcam.cap_transferinfo()
            if info is False:
                return frames

            newest = info.nNewestFrameIndex
            total_count = info.nFrameCount
            n_buf = self.buffer_frames

            if self._last_frame_count < 0:
                # first read after start_freerun(): nothing consumed yet
                delta = min(total_count, n_buf)
            else:
                delta = total_count - self._last_frame_count

            if delta <= 0:
                return frames

            if delta > n_buf:
                lost = delta - n_buf
                self.dropped_frames += lost
                logger.warning(
                    f'DCAM ring buffer overrun: {lost} frame(s) dropped '
                    f'(consumer too slow for buffer_frames={n_buf}). '
                    f'Total dropped this run: {self.dropped_frames}.'
                )
                delta = n_buf  # only the n_buf most recent frames survive

            start = (newest - delta + 1) % n_buf
            idx = start
            for _ in range(delta):
                data = self.dcam.buf_getframedata(idx)
                if data is not False:
                    frames.append(data.copy())
                else:
                    self.dcam.lasterr()._logger_error(
                        logger, f'buf_getframedata({idx}) failed'
                    )
                idx = (idx + 1) % n_buf

            self._last_frame_count = total_count

        return frames

    def snap_image(self) -> np.ndarray:
        with self:
            if not self.dcam.buf_alloc(1):
                self.dcam.lasterr()._logger_error(logger, 'buf_alloc(1) failed')
                return np.zeros((self.height, self.width), dtype=np.uint16)
            try:
                if self.dcam.cap_snapshot():
                    if self.dcam.wait_capevent_frameready(
                        int(max(self.exposure_current * 1000 + 1000, 1000))
                    ):
                        data = self.dcam.buf_getlastframedata()
                        if data is not False:
                            return data
                    self.dcam.lasterr()._logger_error(logger, 'snap timed out')
                else:
                    self.dcam.lasterr()._logger_error(logger, 'cap_snapshot failed')
            finally:
                self.dcam.buf_release()
        return np.zeros((self.height, self.width), dtype=np.uint16)

    # ------------------------------------------------------------------
    # status / property tree
    # ------------------------------------------------------------------
    def populate_status(self):
        with self:
            self.status['Camera'] = {
                'Name': self.name,
                'Serial': getattr(self, 'serial', 'NA'),
            }
            self.status['Exposure'] = {
                'Value': self.get_exposure(),
                'Unit': self.exposure_unit,
            }
            self.status['Temperature'] = {
                'Value': self.get_temperature(),
                'Unit': 'Celsius',
            }

    def _enum_options(self, idprop, mapping: dict[str, int]):
        '''Keep only entries the camera actually reports as valid.'''
        avail = {}
        with self:
            for name, val in mapping.items():
                text = self.dcam.prop_getvaluetext(idprop, float(val))
                if text is not False:
                    avail[name] = val
        return avail or mapping

    def property_tree(self) -> list[dict[str, Any]]:
        """
        Return a list of dicts describing the camera's properties in a
        tree structure suitable for display in a GUI. Each dict has keys:

        - 'name': leaf name of the property (str)
        - 'type': one of 'int', 'float', 'bool', or 'list'
        - 'limits': tuple of (min, max) for 'int' and 'float' types
        - 'value': current value of the property
        - 'enabled': boolean indicating if the property is editable
        - 'parent': parent node in the tree (or None for root nodes)
        """
        props: list[dict[str, Any]] = []

        def can_edit(idprop) -> bool:
            attr = self.dcam.prop_getattr(idprop)
            if attr is False:
                return False
            if self._capturing:
                return attr.is_accessbusy()
            return attr.is_accessready() or attr.is_writable()

        def enum_prop(idprop, name, value_map: dict[str, int], parent):
            attr = self.dcam.prop_getattr(idprop)
            if attr is False:
                return None

            supported = {}
            for label, val in value_map.items():
                if self.dcam.prop_getvaluetext(idprop, float(val)) is not False:
                    supported[label] = val
            if not supported:
                supported = value_map

            current = self.dcam.prop_getvalue(idprop)
            current_label = next(
                (
                    label
                    for label, val in supported.items()
                    if current is not False and int(current) == val
                ),
                next(iter(supported)),
            )
            return {
                'name': name,
                'type': 'list',
                'limits': list(supported.keys()),
                'value': current_label,
                'enabled': can_edit(idprop),
                'parent': parent,
            }

        def int_prop(idprop, name, parent, value_override=None):
            attr = self.dcam.prop_getattr(idprop)
            if attr is False:
                return None
            value = (
                value_override
                if value_override is not None
                else self.dcam.prop_getvalue(idprop)
            )
            return {
                'name': name,
                'type': 'int',
                'limits': (int(attr.valuemin), int(attr.valuemax)),
                'value': int(value if value is not False else attr.valuemin),
                'enabled': can_edit(idprop),
                'parent': parent,
            }

        def bool_prop(idprop, name, parent):
            attr = self.dcam.prop_getattr(idprop)
            if attr is False:
                return None
            value = self.dcam.prop_getvalue(idprop)
            return {
                'name': name,
                'type': 'bool',
                'value': bool(
                    value is not False and int(value) == dcamapi4.DCAMPROP.MODE.ON
                ),
                'enabled': can_edit(idprop),
                'parent': parent,
            }

        def float_prop(idprop, name, parent, readonly=False):
            attr = self.dcam.prop_getattr(idprop)
            if attr is False:
                return None
            value = self.dcam.prop_getvalue(idprop)
            return {
                'name': name,
                'type': 'float',
                'limits': (attr.valuemin, attr.valuemax),
                'value': float(value) if value is not False else attr.valuemin,
                'enabled': can_edit(idprop),
                'readonly': readonly,
                'parent': parent,
            }

        props.extend(
            [
                {
                    'name': str(DcamParams.FREERUN),
                    'type': 'action',
                    'parent': DcamParams.FREERUN.get_parent(),
                },
                {
                    'name': str(DcamParams.STOP),
                    'type': 'action',
                    'parent': DcamParams.STOP.get_parent(),
                },
                {
                    'name': str(DcamParams.BUFFER_FRAMES),
                    'type': 'int',
                    'limits': (4, 512),
                    'value': self.buffer_frames,
                    'enabled': not self._capturing,
                    'parent': DcamParams.BUFFER_FRAMES.get_parent(),
                },
                {
                    'name': str(DcamParams.FRAMERATE),
                    'type': 'float',
                    'limits': (0.0, 100.0),
                    'value': self.framerate_current,
                    'enabled': False,
                    'parent': DcamParams.FRAMERATE.get_parent(),
                },
            ]
        )

        with self:
            for key, val in dcamera._ENUM_CAM_MAP.items():
                idprop, mapping = val
                entry = enum_prop(idprop, str(key), mapping, key.get_parent())
                if entry is not None:
                    props.append(entry)
            for key, val in dcamera._BOOL_CAM_MAP.items():
                idprop, mapping = val
                entry = bool_prop(idprop, str(key), key.get_parent())
                if entry is not None:
                    props.append(entry)
            for key, idprop in dcamera._NUMERIC_CAM_MAP.items():
                entry = int_prop(idprop, str(key), key.get_parent())
                if entry is not None:
                    props.append(entry)
            for key, idprop in dcamera._FLOAT_CAM_MAP.items():
                entry = float_prop(idprop, str(key), key.get_parent())
                if entry is not None:
                    props.append(entry)

        return props

    # leaf name -> (DCAM_IDPROP id, {label: value} enum map or None for numeric)
    _ENUM_CAM_MAP = {
        DcamParams.READOUT_SPEED: (
            dcamapi4.DCAM_IDPROP.READOUTSPEED,
            {
                'Ultra-Quiet / Slow Scan': dcamapi4.DCAMPROP.READOUTSPEED.SLOWEST,
                'Standard Scan': 2,
                'Fast Scan': 3,
                'Fastest Scan': dcamapi4.DCAMPROP.READOUTSPEED.FASTEST,
            },
        ),
        DcamParams.BITS_PER_CHANNEL: (
            dcamapi4.DCAM_IDPROP.BITSPERCHANNEL,
            {
                '8': dcamapi4.DCAMPROP.BITSPERCHANNEL._8,
                '10': dcamapi4.DCAMPROP.BITSPERCHANNEL._10,
                '12': dcamapi4.DCAMPROP.BITSPERCHANNEL._12,
                '14': dcamapi4.DCAMPROP.BITSPERCHANNEL._14,
                '16': dcamapi4.DCAMPROP.BITSPERCHANNEL._16,
            },
        ),
        DcamParams.COOLER_FAN: (
            dcamapi4.DCAM_IDPROP.SENSORCOOLERFAN,
            {
                'Off': dcamapi4.DCAMPROP.MODE.OFF,
                'On': dcamapi4.DCAMPROP.MODE.ON,
            },
        ),
        DcamParams.SENSOR_MODE: (
            dcamapi4.DCAM_IDPROP.SENSORMODE,
            {
                'Area': dcamapi4.DCAMPROP.SENSORMODE.AREA,
                'Line': dcamapi4.DCAMPROP.SENSORMODE.LINE,
                'Partial Area': dcamapi4.DCAMPROP.SENSORMODE.PARTIALAREA,
                'Progressive': dcamapi4.DCAMPROP.SENSORMODE.PROGRESSIVE,
            },
        ),
        DcamParams.BINNING: (
            dcamapi4.DCAM_IDPROP.BINNING,
            {
                '1x1': dcamapi4.DCAMPROP.BINNING._1,
                '2x2': dcamapi4.DCAMPROP.BINNING._2,
                '4x4': dcamapi4.DCAMPROP.BINNING._4,
                '8x8': dcamapi4.DCAMPROP.BINNING._8,
                '16x16': dcamapi4.DCAMPROP.BINNING._16,
                '1x2': dcamapi4.DCAMPROP.BINNING._1_2,
                '2x4': dcamapi4.DCAMPROP.BINNING._2_4,
            },
        ),
        DcamParams.SHUTTER_MODE: (
            dcamapi4.DCAM_IDPROP.SHUTTER_MODE,
            {
                'Global': dcamapi4.DCAMPROP.SHUTTER_MODE.GLOBAL,
                'Rolling': dcamapi4.DCAMPROP.SHUTTER_MODE.ROLLING,
            },
        ),
        DcamParams.TRIGGER_SOURCE: (
            dcamapi4.DCAM_IDPROP.TRIGGERSOURCE,
            {
                'Internal': dcamapi4.DCAMPROP.TRIGGERSOURCE.INTERNAL,
                'External': dcamapi4.DCAMPROP.TRIGGERSOURCE.EXTERNAL,
                'Software': dcamapi4.DCAMPROP.TRIGGERSOURCE.SOFTWARE,
                'Master Pulse': dcamapi4.DCAMPROP.TRIGGERSOURCE.MASTERPULSE,
            },
        ),
        DcamParams.TRIGGER_MODE: (
            dcamapi4.DCAM_IDPROP.TRIGGER_MODE,
            {
                'Normal': dcamapi4.DCAMPROP.TRIGGER_MODE.NORMAL,
                'PIV': dcamapi4.DCAMPROP.TRIGGER_MODE.PIV,
                'Start': dcamapi4.DCAMPROP.TRIGGER_MODE.START,
            },
        ),
        DcamParams.TRIGGER_ACTIVE: (
            dcamapi4.DCAM_IDPROP.TRIGGERACTIVE,
            {
                'Edge': dcamapi4.DCAMPROP.TRIGGERACTIVE.EDGE,
                'Level': dcamapi4.DCAMPROP.TRIGGERACTIVE.LEVEL,
                'Sync Readout': dcamapi4.DCAMPROP.TRIGGERACTIVE.SYNCREADOUT,
                'Point': dcamapi4.DCAMPROP.TRIGGERACTIVE.POINT,
            },
        ),
        DcamParams.SENSOR_COOLER: (
            dcamapi4.DCAM_IDPROP.SENSORCOOLER,
            {
                'Off': dcamapi4.DCAMPROP.SENSORCOOLER.OFF,
                'On': dcamapi4.DCAMPROP.SENSORCOOLER.ON,
                'Max': dcamapi4.DCAMPROP.SENSORCOOLER.MAX,
            },
        ),
        DcamParams.SUBARRAY_MODE: (
            dcamapi4.DCAM_IDPROP.SUBARRAYMODE,
            {'Off': dcamapi4.DCAMPROP.MODE.OFF, 'On': dcamapi4.DCAMPROP.MODE.ON},
        ),
        DcamParams.DEFECTCORRECT_MODE: (
            dcamapi4.DCAM_IDPROP.DEFECTCORRECT_MODE,
            {
                'Off': dcamapi4.DCAMPROP.DEFECTCORRECT_MODE.OFF,
                'On': dcamapi4.DCAMPROP.DEFECTCORRECT_MODE.ON,
            },
        ),
        DcamParams.DEFECTCORRECT_METHOD: (
            dcamapi4.DCAM_IDPROP.DEFECTCORRECT_METHOD,
            {
                'Ceiling': dcamapi4.DCAMPROP.DEFECTCORRECT_METHOD.CEILING,
                'Previous': dcamapi4.DCAMPROP.DEFECTCORRECT_METHOD.PREVIOUS,
                'Next': dcamapi4.DCAMPROP.DEFECTCORRECT_METHOD.NEXT,
            },
        ),
        DcamParams.HOTPIXELCORRECT_LEVEL: (
            dcamapi4.DCAM_IDPROP.HOTPIXELCORRECT_LEVEL,
            {
                'Aggressive': dcamapi4.DCAMPROP.HOTPIXELCORRECT_LEVEL.AGGRESSIVE,
                'Minimum': dcamapi4.DCAMPROP.HOTPIXELCORRECT_LEVEL.MINIMUM,
                'Standard': dcamapi4.DCAMPROP.HOTPIXELCORRECT_LEVEL.STANDARD,
            },
        ),
        DcamParams.GPIO_CHANNELSYNC: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_CHANNELSYNC,
            {
                '1 Channel': dcamapi4.DCAMPROP.OUTPUTTRIGGER_CHANNELSYNC._1CHANNEL,
                '2 Channels': dcamapi4.DCAMPROP.OUTPUTTRIGGER_CHANNELSYNC._2CHANNELS,
                '3 Channels': dcamapi4.DCAMPROP.OUTPUTTRIGGER_CHANNELSYNC._3CHANNELS,
            },
        ),
        DcamParams.GPIO_1_ACTIVE: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_ACTIVE,
            {
                'Edge': dcamapi4.DCAMPROP.OUTPUTTRIGGER_ACTIVE.EDGE,
                'Level': dcamapi4.DCAMPROP.OUTPUTTRIGGER_ACTIVE.LEVEL,
            },
        ),
        DcamParams.GPIO_1_SOURCE: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_SOURCE,
            {
                'Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.EXPOSURE,
                'Readout End': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.READOUTEND,
                'VSync': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.VSYNC,
                'HSync': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.HSYNC,
                'Trigger': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.TRIGGER,
            },
        ),
        DcamParams.GPIO_1_POLARITY: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_POLARITY,
            {
                'Positive': dcamapi4.DCAMPROP.OUTPUTTRIGGER_POLARITY.POSITIVE,
                'Negative': dcamapi4.DCAMPROP.OUTPUTTRIGGER_POLARITY.NEGATIVE,
            },
        ),
        DcamParams.GPIO_1_KIND: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_KIND,
            {
                'Low': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.LOW,
                'Global Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,
                'Programable': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.PROGRAMABLE,
                'Trigger Ready': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.TRIGGERREADY,
                'High': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.HIGH,
                'Any Row Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.ANYROWEXPOSURE,
            },
        ),
        DcamParams.GPIO_1_PROGRAMABLESTART: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_PROGRAMABLESTART,
            {
                'First Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_PRSTART.FIRSTEXPOSURE,
                'First Readout': dcamapi4.DCAMPROP.OUTPUTTRIGGER_PRSTART.FIRSTREADOUT,
            },
        ),
        DcamParams.GPIO_1_BASESENSOR: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_BASESENSOR,
            {
                'View 1': dcamapi4.DCAMPROP.OUTPUTTRIGGER_BASESENSOR.VIEW1,
                'View 2': dcamapi4.DCAMPROP.OUTPUTTRIGGER_BASESENSOR.VIEW2,
                'Any View': dcamapi4.DCAMPROP.OUTPUTTRIGGER_BASESENSOR.ANYVIEW,
                'All Views': dcamapi4.DCAMPROP.OUTPUTTRIGGER_BASESENSOR.ALLVIEWS,
            },
        ),
        DcamParams.GPIO_2_ACTIVE: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_ACTIVE + 0x100,
            {
                'Edge': dcamapi4.DCAMPROP.OUTPUTTRIGGER_ACTIVE.EDGE,
                'Level': dcamapi4.DCAMPROP.OUTPUTTRIGGER_ACTIVE.LEVEL,
            },
        ),
        DcamParams.GPIO_2_SOURCE: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_SOURCE + 0x100,
            {
                'Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.EXPOSURE,
                'Readout End': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.READOUTEND,
                'VSync': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.VSYNC,
                'HSync': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.HSYNC,
                'Trigger': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.TRIGGER,
            },
        ),
        DcamParams.GPIO_2_POLARITY: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_POLARITY + 0x100,
            {
                'Positive': dcamapi4.DCAMPROP.OUTPUTTRIGGER_POLARITY.POSITIVE,
                'Negative': dcamapi4.DCAMPROP.OUTPUTTRIGGER_POLARITY.NEGATIVE,
            },
        ),
        DcamParams.GPIO_2_KIND: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_KIND + 0x100,
            {
                'Low': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.LOW,
                'Global Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,
                'Programable': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.PROGRAMABLE,
                'Trigger Ready': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.TRIGGERREADY,
                'High': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.HIGH,
                'Any Row Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.ANYROWEXPOSURE,
            },
        ),
        DcamParams.GPIO_2_PROGRAMABLESTART: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_PROGRAMABLESTART + 0x100,
            {
                'First Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_PRSTART.FIRSTEXPOSURE,
                'First Readout': dcamapi4.DCAMPROP.OUTPUTTRIGGER_PRSTART.FIRSTREADOUT,
            },
        ),
        DcamParams.GPIO_3_ACTIVE: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_ACTIVE + 0x200,
            {
                'Edge': dcamapi4.DCAMPROP.OUTPUTTRIGGER_ACTIVE.EDGE,
                'Level': dcamapi4.DCAMPROP.OUTPUTTRIGGER_ACTIVE.LEVEL,
            },
        ),
        DcamParams.GPIO_3_SOURCE: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_SOURCE + 0x200,
            {
                'Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.EXPOSURE,
                'Readout End': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.READOUTEND,
                'VSync': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.VSYNC,
                'HSync': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.HSYNC,
                'Trigger': dcamapi4.DCAMPROP.OUTPUTTRIGGER_SOURCE.TRIGGER,
            },
        ),
        DcamParams.GPIO_3_POLARITY: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_POLARITY + 0x200,
            {
                'Positive': dcamapi4.DCAMPROP.OUTPUTTRIGGER_POLARITY.POSITIVE,
                'Negative': dcamapi4.DCAMPROP.OUTPUTTRIGGER_POLARITY.NEGATIVE,
            },
        ),
        DcamParams.GPIO_3_KIND: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_KIND + 0x200,
            {
                'Low': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.LOW,
                'Global Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,
                'Programable': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.PROGRAMABLE,
                'Trigger Ready': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.TRIGGERREADY,
                'High': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.HIGH,
                'Any Row Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_KIND.ANYROWEXPOSURE,
            },
        ),
        DcamParams.GPIO_3_PROGRAMABLESTART: (
            dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_PROGRAMABLESTART + 0x200,
            {
                'First Exposure': dcamapi4.DCAMPROP.OUTPUTTRIGGER_PRSTART.FIRSTEXPOSURE,
                'First Readout': dcamapi4.DCAMPROP.OUTPUTTRIGGER_PRSTART.FIRSTREADOUT,
            },
        ),
    }

    _BOOL_CAM_MAP: dict[DcamParams, int] = {}

    _NUMERIC_CAM_MAP = {
        DcamParams.GPIOCONNECTORS: dcamapi4.DCAM_IDPROP.NUMBEROF_OUTPUTTRIGGERCONNECTOR,
    }

    DCAMPROP_READOUTSPEED__SLOWEST = dcamapi4.DCAMPROP.READOUTSPEED.SLOWEST

    _FLOAT_CAM_MAP = {
        DcamParams.TEMPERATURE_TARGET: dcamapi4.DCAM_IDPROP.SENSORTEMPERATURETARGET,
        DcamParams.CONVERSIONFACTOR_COEFF: dcamapi4.DCAM_IDPROP.CONVERSIONFACTOR_COEFF,
        DcamParams.GPIO_1_DELAY: dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_DELAY,
        DcamParams.GPIO_1_PERIOD: dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_PERIOD,
        DcamParams.GPIO_2_DELAY: dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_DELAY + 0x100,
        DcamParams.GPIO_2_PERIOD: dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_PERIOD + 0x100,
        DcamParams.GPIO_3_DELAY: dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_DELAY + 0x200,
        DcamParams.GPIO_3_PERIOD: dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_PERIOD + 0x200,
    }

    def update_cam(self, param, path: Optional[list[str]], param_value: Any):
        '''
        Route a pyqtgraph parameter change back to the DCAM property it
        represents. Every editable leaf added in property_tree() must have
        a matching branch here (or a subclass override) or the GUI control
        silently stops doing anything when touched — that mismatch is what
        happened before this pass, so the three maps above are kept next to
        this method deliberately, to make it obvious when they drift apart.
        '''
        if not path:
            logger.debug('update_cam: no path provided')
            return

        try:
            dcam_param = DcamParams('.'.join(path))
        except ValueError:
            return

        # checkif param_value is scalar (int, float, bool, str) if not return
        if not isinstance(param_value, (int, float, bool, str)):
            return

        with self:
            if dcam_param == DcamParams.BUFFER_FRAMES:
                self.buffer_frames = int(param_value)
                return

            if dcam_param in self._ENUM_CAM_MAP:
                idprop, mapping = self._ENUM_CAM_MAP[dcam_param]
                if param_value not in mapping:
                    logger.warning(
                        f'Unknown {str(dcam_param)!r} value: {param_value!r}'
                    )
                    return
                if not self.dcam.prop_setvalue(idprop, mapping[param_value]):
                    self.dcam.lasterr()._logger_error(
                        logger, f'set {str(dcam_param)!r} failed'
                    )
                return

            if dcam_param in self._BOOL_CAM_MAP:
                idprop = self._BOOL_CAM_MAP[dcam_param]
                mode = (
                    dcamapi4.DCAMPROP.MODE.ON
                    if param_value
                    else dcamapi4.DCAMPROP.MODE.OFF
                )
                if not self.dcam.prop_setvalue(idprop, mode):
                    self.dcam.lasterr()._logger_error(
                        logger, f'set {str(dcam_param)!r} failed'
                    )
                return

            if dcam_param in self._NUMERIC_CAM_MAP:
                idprop = self._NUMERIC_CAM_MAP[dcam_param]
                if not self.dcam.prop_setvalue(idprop, int(param_value)):
                    self.dcam.lasterr()._logger_error(
                        logger, f'set {str(dcam_param)!r} failed'
                    )
                return

            if dcam_param in self._FLOAT_CAM_MAP:
                idprop = self._FLOAT_CAM_MAP[dcam_param]
                if not self.dcam.prop_setvalue(idprop, float(param_value)):
                    self.dcam.lasterr()._logger_error(
                        logger, f'set {str(dcam_param)!r} failed'
                    )
                return

            logger.debug(f'update_cam: no handler for leaf {str(dcam_param)!r}')
