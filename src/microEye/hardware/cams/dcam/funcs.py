import logging

import cv2

from microEye.hardware.cams.dcam.dcam import Dcam, Dcamapi, dcamapi4, np

logger = logging.getLogger(__name__)


def dcam_show_device_list():
    '''Show device list.

    Show model names ans IDs of the connected camera.

    Returns:
        Nothing.
    '''
    if Dcamapi.init():
        n = Dcamapi.get_devicecount()
        for i in range(0, n):
            dcam = Dcam(i)
            output = f'#{i}: '

            model = dcam.dev_getstring(dcamapi4.DCAM_IDSTR.MODEL)
            if model is False:
                output = output + 'No DCAM_IDSTR.MODEL'
            else:
                output = output + f'MODEL={model}'

            cameraid = dcam.dev_getstring(dcamapi4.DCAM_IDSTR.CAMERAID)
            if cameraid is False:
                output = output + ', No DCAM_IDSTR.CAMERAID'
            else:
                output = output + f', CAMERAID={cameraid}'

            logger.info(output)
    elif Dcamapi.lasterr() == dcamapi4.DCAMERR.NOCAMERA:
        logger.warning('No camera found.')
    else:
        Dcamapi.lasterr()._logger_error(logger, 'Dcamapi.init() failed with error')

    Dcamapi.uninit()


def dcam_show_properties(iDevice=0):
    '''
    Show the list of properties.

    Show the list of property IDs and values that the camera supported.

    Args:
        iDevice (int): Device index.

    Returns:
        Nothing.
    '''
    if Dcamapi.init():
        dcam = Dcam(iDevice)
        if dcam.dev_open():
            idprop = dcam.prop_getnextid(0)
            while idprop is not False:
                output = f'0x{idprop:08X}: '

                propname = dcam.prop_getname(idprop)
                if propname is not False:
                    output = output + propname

                logger.info(output)
                idprop = dcam.prop_getnextid(idprop)

            dcam.dev_close()
        else:
            dcam.lasterr()._logger_error(logger, 'Dcam.dev_open() failed with error')
    else:
        Dcamapi.lasterr()._logger_error(logger, 'Dcamapi.init() failed with error')

    Dcamapi.uninit()


def dcamtest_show_framedata(data: np.ndarray, windowtitle: str, windowstatus: int):
    '''
    Show numpy buffer as an image with OpenCV function.

    Parameters
    ----------
    data : np.ndarray
        NumPy array.
    windowtitle : str
        Window name.
    windowstatus : int
        Last window status returned by dcamtest_show_framedata function.
        Specify 0 when calling the first time.

    Returns
    -------
    int
        Window status.
    '''
    if (
        windowstatus > 0
        and cv2.getWindowProperty(windowtitle, cv2.WND_PROP_VISIBLE) == 0
    ):
        return -1
        # Window has been closed.
    if windowstatus < 0:
        return -1
        # Window is already closed.

    if data.dtype == np.uint16:
        imax = np.amax(data)
        if imax > 0:
            imul = int(65535 / imax)
            # print('Multiple %s' % imul)
            data = data * imul

        cv2.imshow(windowtitle, data)
        return 1
    else:
        print('-NG: dcamtest_show_image(data) only support Numpy.uint16 data')
        return -1


def dcam_show_single_captured_image(iDevice=0):
    '''Show a single image.

    Capture and show a single image.

    Args:
        iDevice (int): Device index.

    Returns:
        Nothing.
    '''
    if Dcamapi.init():
        dcam = Dcam(iDevice)
        if dcam.dev_open():
            if dcam.buf_alloc(1):
                if dcam.cap_snapshot():
                    timeout_milisec = 1000
                    while True:
                        if dcam.wait_capevent_frameready(timeout_milisec):
                            data = dcam.buf_getlastframedata()
                            dcamtest_show_framedata(data)
                            break

                        dcamerr = dcam.lasterr()
                        if dcamerr.is_timeout():
                            logger.error('===: timeout')
                            continue

                        dcamerr._logger_error(
                            logger, 'Dcam.wait_event() fails with error'
                        )
                        break
                else:
                    dcam.lasterr()._logger_error(
                        logger, 'Dcam.cap_start() fails with error'
                    )

                dcam.buf_release()
            else:
                dcam.lasterr()._logger_error(
                    logger, 'Dcam.buf_alloc(1) fails with error'
                )
            dcam.dev_close()
        else:
            dcam.lasterr()._logger_error(logger, 'Dcam.dev_open() fails with error')
    else:
        Dcamapi.lasterr()._logger_error(logger, 'Dcamapi.init() fails with error')

    Dcamapi.uninit()


def dcamtest_thread_live(dcam: Dcam):
    '''Show live images.

    Capture and show live images.

    Args:
        dcam (Dcam): Dcam instance.

    Returns:
        Nothing.
    '''
    if dcam.cap_start():
        timeout_milisec = 100
        iWindowStatus = 0
        while iWindowStatus >= 0:
            if dcam.wait_capevent_frameready(timeout_milisec):
                data = dcam.buf_getlastframedata()
                iWindowStatus = dcamtest_show_framedata(data, 'test', iWindowStatus)
            else:
                dcamerr = dcam.lasterr()
                if dcamerr.is_timeout():
                    logger.error('===: timeout')
                else:
                    dcamerr._logger_error(logger, 'Dcam.wait_event() fails with error')
                    break

            key = cv2.waitKey(1)
            if key == ord('q') or key == ord(
                'Q'
            ):  # if 'q' was pressed with the live window, close it
                break

        dcam.cap_stop()
    else:
        dcam.lasterr()._logger_error(logger, 'Dcam.cap_start() fails with error')


def dcam_live_capturing(iDevice=0):
    '''Capture live images.

    Recognize camera and capture live images.

    Args:
        iDevice (int): Device index.

    Returns:
        Nothing.
    '''
    if Dcamapi.init():
        dcam = Dcam(iDevice)
        if dcam.dev_open():
            if dcam.buf_alloc(3):
                # th = threading.Thread(target=dcamtest_thread_live, args=(dcam,))
                # th.start()
                # th.join()
                dcamtest_thread_live(dcam)

                # release buffer
                dcam.buf_release()
            else:
                dcam.lasterr()._logger_error(
                    logger, 'Dcam.buf_alloc(3) fails with error'
                )
            dcam.dev_close()
        else:
            dcam.lasterr()._logger_error(logger, 'Dcam.dev_open() fails with error')
    else:
        dcam.lasterr()._logger_error(logger, 'Dcamapi.init() fails with error')

    Dcamapi.uninit()


if __name__ == '__main__':
    dcam_show_device_list()
