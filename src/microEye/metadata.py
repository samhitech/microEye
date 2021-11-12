from tkinter import Widget
import ome_types.model as om
import typing

from ome_types.model.channel import *
from ome_types.model.ome import OME
from ome_types.model.simple_types import UnitsLength, UnitsTime
from ome_types.model.tiff_data import TiffData

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pydantic.utils import ClassAttribute


class MetadataEditor(QScrollArea):

    def __init__(self, parent: typing.Optional['QWidget'] = None):
        super().__init__(parent=parent)

        self.InitLayout()

    def InitLayout(self):

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)

        self.Widget = QWidget()

        # main layout
        self.mainLayout = QVBoxLayout()
        self.Widget.setLayout(self.mainLayout)

        self.dValid = QDoubleValidator()
        self.iValid = QIntValidator()

        # self.mainLayout.addWidget(QLabel(''))
        # self.mainLayout.addWidget()

        self.experiment = QLineEdit('Experiment_001')
        self.mainLayout.addWidget(QLabel('Experiment name:'))
        self.mainLayout.addWidget(self.experiment)

        self.exp_desc = QTextEdit('')
        self.mainLayout.addWidget(QLabel('Experiment description:'))
        self.mainLayout.addWidget(self.exp_desc)

        self.exp_email = QLineEdit()
        self.exp_fName = QLineEdit()
        self.exp_lName = QLineEdit()
        self.exp_institute = QLineEdit()

        self.mainLayout.addWidget(QLabel('Experimenter First name:'))
        self.mainLayout.addWidget(self.exp_fName)
        self.mainLayout.addWidget(QLabel('Experimenter Last name:'))
        self.mainLayout.addWidget(self.exp_lName)
        self.mainLayout.addWidget(QLabel('Experimenter e-mail:'))
        self.mainLayout.addWidget(self.exp_email)
        self.mainLayout.addWidget(QLabel('Experimenter Institute:'))
        self.mainLayout.addWidget(self.exp_institute)

        self.px_size = QLineEdit('120')
        self.px_size.setValidator(self.dValid)
        self.py_size = QLineEdit('120')
        self.py_size.setValidator(self.dValid)
        self.p_unit = QComboBox()
        self.p_unit.addItems(UnitsLength._member_names_)
        self.p_unit.setCurrentText(UnitsLength.NANOMETER.name)

        self.mainLayout.addWidget(QLabel('Pixel x-size:'))
        self.mainLayout.addWidget(self.px_size)
        self.mainLayout.addWidget(QLabel('Pixel y-size:'))
        self.mainLayout.addWidget(self.py_size)
        self.mainLayout.addWidget(QLabel('Unit:'))
        self.mainLayout.addWidget(self.p_unit)

        self.channel_name = QLineEdit('CAM_1')
        self.fluor_name = QLineEdit()

        self.mainLayout.addWidget(QLabel('Channel name:'))
        self.mainLayout.addWidget(self.channel_name)
        self.mainLayout.addWidget(QLabel('Fluorophore:'))
        self.mainLayout.addWidget(self.fluor_name)

        self.exposure = QLineEdit('100.0')
        self.exposure.setValidator(self.dValid)
        self.mainLayout.addWidget(QLabel('Exposure time (ms):'))
        self.mainLayout.addWidget(self.exposure)

        self.acq_mode = QComboBox()
        self.acq_mode.addItems(AcquisitionMode._member_names_)
        self.acq_mode.setCurrentText(AcquisitionMode.TIRF.name)

        self.mainLayout.addWidget(QLabel('Acquisition mode:'))
        self.mainLayout.addWidget(self.acq_mode)

        self.ill_type = QComboBox()
        self.ill_type.addItems(IlluminationType._member_names_)
        self.ill_type.setCurrentText(IlluminationType.OTHER.name)

        self.mainLayout.addWidget(QLabel('Illumination type:'))
        self.mainLayout.addWidget(self.ill_type)

        self.contrast = QComboBox()
        self.contrast.addItems(ContrastMethod._member_names_)
        self.contrast.setCurrentText(ContrastMethod.FLUORESCENCE.name)

        self.mainLayout.addWidget(QLabel('Contrast method:'))
        self.mainLayout.addWidget(self.contrast)

        self.emission = QLineEdit('0.0')
        self.emission.setValidator(self.dValid)
        self.excitation = QLineEdit('638')
        self.excitation.setValidator(self.dValid)
        self.wave_unit = QComboBox()
        self.wave_unit.addItems(UnitsLength._member_names_)
        self.wave_unit.setCurrentText(UnitsLength.NANOMETER.name)

        self.mainLayout.addWidget(QLabel('Excitation wavelength:'))
        self.mainLayout.addWidget(self.excitation)
        self.mainLayout.addWidget(QLabel('Emission wavelength:'))
        self.mainLayout.addWidget(self.emission)
        self.mainLayout.addWidget(QLabel('Wavelength unit:'))
        self.mainLayout.addWidget(self.wave_unit)

        self.mainLayout.addWidget(QPushButton(
            'Generate',
            clicked=lambda: self.getOME_XML(100, 256, 512)))

        self.setWidget(self.Widget)

    def getOME_XML(self, frames: int, width: int, height: int):
        ome_obj = om.OME(creator='microEye Python Package')

        experimenter = om.Experimenter()
        experimenter.first_name = self.exp_fName.text()
        experimenter.last_name = self.exp_lName.text()
        experimenter.email = self.exp_email.text()
        experimenter.institution = self.exp_institute.text()
        ome_obj.experimenters.append(experimenter)

        planes = [om.Plane(
            the_c=0,
            the_t=i,
            the_z=0,
            exposure_time=float(self.exposure.text()),
            exposure_time_unit=UnitsTime.MILLISECOND
        ) for i in range(frames)]

        channel = om.Channel()
        channel.name = self.channel_name.text()
        channel.fluor = self.fluor_name.text()
        channel.acquisition_mode = AcquisitionMode._member_map_[
            self.acq_mode.currentText()]
        channel.illumination_type = IlluminationType._member_map_[
            self.ill_type.currentText()]
        channel.contrast_method = ContrastMethod._member_map_[
            self.contrast.currentText()]
        channel.excitation_wavelength = float(self.excitation.text())
        channel.emission_wavelength = float(self.emission.text())
        channel.excitation_wavelength_unit = UnitsLength._member_map_[
            self.wave_unit.currentText()]
        channel.emission_wavelength_unit = UnitsLength._member_map_[
            self.wave_unit.currentText()]
        channel.samples_per_pixel = 1

        pixels = om.Pixels(
            size_c=1, size_t=frames,
            size_x=width,
            size_y=height, size_z=1,
            type='uint16', dimension_order='XYZCT',
            # metadata_only=True,
            physical_size_x=float(self.px_size.text()),
            physical_size_x_unit=UnitsLength._member_map_[
                self.p_unit.currentText()],
            physical_size_y=float(self.py_size.text()),
            physical_size_y_unit=UnitsLength._member_map_[
                self.p_unit.currentText()],
            time_increment=float(self.exposure.text()),
            time_increment_unit=UnitsTime.MILLISECOND,
        )
        pixels.tiff_data_blocks.append(om.TiffData())
        pixels.channels.append(channel)
        pixels.planes.extend(planes)
        img = om.Image(
            id='Image:1',
            name=self.experiment.text(),
            pixels=pixels,
            description=self.exp_desc.toPlainText()
        )
        ome_obj.images.append(img)
        ome_obj = OME.validate(ome_obj)

        return ome_obj
        # tf.tiffcomment(self.tiff.filename, ome.to_xml())
