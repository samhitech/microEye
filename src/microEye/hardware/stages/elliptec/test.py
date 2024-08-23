import sys
import time
from enum import Enum, auto

from microEye.hardware.stages.elliptec.baseDevice import BaseDevice, DeviceDirection
from microEye.hardware.stages.elliptec.deviceID import DeviceID
from microEye.hardware.stages.elliptec.devicePort import DevicePort
from microEye.hardware.stages.elliptec.ellDevices import ELLDevices
from microEye.qt import QApplication


def data_sent_handler(message: str):
    print(f'\033[96mTx: {message}\033[0m')  # Cyan color


def data_received_handler(message: str):
    print(f'\033[92mRx: {message}\033[0m')  # Green color


def output_update_handler(messages: list[str], error: bool):
    for message in messages:
        print(f'\033[93mOutput-> {message}\033[0m')  # Yellow color


def main(args: list[str]):
    # Get the communication port
    port = args[0] if len(args) > 0 else 'COM3'

    # Get the range of addresses used (max range is '0' to 'F')
    min_search_limit = (
        args[1][0].upper()
        if len(args) > 1 and BaseDevice.is_valid_address(args[1][0].upper())
        else '0'
    )
    max_search_limit = (
        args[2][0].upper()
        if len(args) > 2 and BaseDevice.is_valid_address(args[2][0].upper())
        else '3'
    )

    # Setup handlers
    DevicePort.instance().dataSent.connect(data_sent_handler)
    DevicePort.instance().dataReceived.connect(data_received_handler)

    # Create ELLDevices class to maintain the collection of Elliptec devices
    ell_devices = ELLDevices()
    ell_devices.MessageUpdates.outputUpdated.connect(output_update_handler)

    # Attempt to connect to the port
    if DevicePort.open(port):
        print('Discover devices')
        print('================')
        # Scan the port for connected devices using the given range of addresses
        devices = ell_devices.scanAddresses(min_search_limit, max_search_limit)

        for device in devices:
            # Configure each device found
            if ell_devices.configure(device):
                # Test each device found
                print(f'\nIdentify device {device[0]}')
                print('=================')
                addressed_device = ell_devices.addressedDevice(device[0])

                if addressed_device:
                    # Display the device information
                    device_info = addressed_device.deviceInfo
                    for info in device_info.description():
                        print(info)

                    # Test each device according to type
                    print(f'\nTest device {device[0]}')
                    print('=============')
                    if device_info.DeviceType in [
                        DeviceID.DeviceTypes.Shutter2,
                        DeviceID.DeviceTypes.Shutter4,
                    ]:
                        # Test the shutter device
                        addressed_device.home(DeviceDirection.Linear)
                        time.sleep(0.25)
                        addressed_device.jog_forward()
                        time.sleep(0.25)
                        if device_info.DeviceType == DeviceID.DeviceTypes.Shutter4:
                            addressed_device.jog_forward()
                            time.sleep(0.25)
                            addressed_device.jog_forward()
                            time.sleep(0.25)
                        addressed_device.jog_backward()
                        time.sleep(0.25)
                        if device_info.DeviceType == DeviceID.DeviceTypes.Shutter4:
                            addressed_device.jog_backward()
                            time.sleep(0.25)
                            addressed_device.jog_backward()
                            time.sleep(0.25)
                    elif device_info.DeviceType in [
                        DeviceID.DeviceTypes.LinearStage25mm,
                        DeviceID.DeviceTypes.LinearStage28mm,
                        DeviceID.DeviceTypes.LinearStage60mm,
                        DeviceID.DeviceTypes.LinearStage60mm_10,
                    ]:
                        # Test the Linear stage
                        # For each motor ('1' and '2') get the motor information
                        for c in ['1', '2']:
                            if addressed_device.get_motor_info(c):
                                motor_info = addressed_device._motor_info[c]
                                for info in motor_info.description():
                                    print(f'Output-> {info}')

                        # Test the stage movement
                        addressed_device.home(DeviceDirection.Linear)
                        time.sleep(0.25)
                        addressed_device.set_jogstep_size(1.0)
                        for _ in range(10):
                            addressed_device.jog_forward()
                            time.sleep(0.1)
                    elif device_info.DeviceType in [DeviceID.DeviceTypes.OpticsRotator]:
                        for c in ['1', '2']:
                            if addressed_device.get_motor_info(c):
                                motor_info = addressed_device._motor_info[c]
                                for info in motor_info.description():
                                    print(f'Output-> {info}')
                        # Test the stage movement
                        addressed_device.home(DeviceDirection.Clockwise)
                        time.sleep(0.25)
                        addressed_device.set_jogstep_size(10.0)
                        for _ in range(10):
                            addressed_device.jog_forward()
                            time.sleep(0.1)

        DevicePort.close()
    else:
        print(f'\033[91mPort {port} unavailable\033[0m')  # Red color

    print('Press Enter to exit')
    input()


if __name__ == '__main__':
    app = QApplication([])
    main(sys.argv[1:])
