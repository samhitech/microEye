import ctypes


class miCamera:
    def __init__(self, Cam_ID=0) -> None:
        self.Cam_ID = Cam_ID
        self.acquisition = False
        self.bytes_per_pixel = 1
        self.exposure_current = 0.0
        self.height = 512
        self.temperature = -127
        self.width = 512
        self.name = ''

    def get_temperature(self):
        return self.temperature

    def getWidth(self):
        if isinstance(self.width, ctypes.c_int):
            return self.width.value
        elif isinstance(self.width, int):
            return self.width

    def getHeight(self):
        if isinstance(self.height, ctypes.c_int):
            return self.height.value
        elif isinstance(self.height, int):
            return self.height

    def getExposure(self):
        if isinstance(self.exposure_current, ctypes.c_double):
            return self.exposure_current.value
        elif isinstance(self.exposure_current, (int, float)):
            return self.exposure_current

