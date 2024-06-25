import hid


class hidDevice:
    def __init__(self, device: dict) -> None:
        '''
        Initialize a new hidDevice object.

        Parameters
        ----------
        device : dict
            A dictionary containing information about the HID device.
        '''
        self.data = device

    @property
    def vendorID(self):
        '''
        Get the vendor ID of the HID device.

        Returns
        -------
        int or None
            The vendor ID of the HID device, or None if it is not available.
        '''
        return self.data.get('vendor_id', None)

    @property
    def productID(self):
        '''
        Get the product ID of the HID device.

        Returns
        -------
        int or None
            The product ID of the HID device, or None if it is not available.
        '''
        return self.data.get('product_id', None)

    def getHID(self):
        '''
        Get a hid.device object for the HID device.

        Returns
        -------
        hid.device
            A hid.device object for the HID device.
        '''
        hid_device = hid.device()
        hid_device.open(self.vendorID, self.productID)
        hid_device.set_nonblocking(True)
        return hid_device

    def __str__(self) -> str:
        '''
        Get a string representation of the HID device.

        Returns
        -------
        str
            A string representation of the HID device.
        '''
        return self.data.get('product_string', 'N/A')
