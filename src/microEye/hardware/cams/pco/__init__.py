import logging
import sys

import pco
from pco.sdk import Sdk, shared_library_loader

interfaces = [
    'FireWire',
    'Camera Link MTX',
    'GenICam',
    'Camera Link NAT',
    'GigE',
    'USB 2.0',
    'Camera Link ME4',
    'USB 3.0',
    'CLHS',
]


def _scanner(interfaces: list[str]):
    shared_library_loader._load_all()
    sdk = Sdk()
    if sys.platform.startswith('linux'):
        ret = sdk.scan_cameras()
        devices_info = ret['devices']
    else:
        devices_info = []
        for interface in interfaces:
            for camera_number in range(64):
                ret = sdk.open_camera_ex(
                    interface=interface, camera_number=camera_number
                )
                if sdk.camera_handle.value is not None:
                    ret['camera name'] = sdk.get_camera_name()['camera name'].replace(
                        ' ', '_'
                    )
                    ret['serial number'] = sdk.get_camera_type()['serial number']
                    ret['interface'] = interface
                    sdk.close_camera()

                if ret['error'] == 0:
                    devices_info.append(ret)
                elif ret['error'] & 0x80002001 == 0x80002001:
                    continue
                else:
                    break

    shared_library_loader.decrement()

    return devices_info


logger = logging.getLogger('pco')
logger.setLevel(logging.DEBUG)
logger.addHandler(pco.stream_handler)


print(_scanner(interfaces))
