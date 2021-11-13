from tkinter import Widget
from numba.cuda.simulator.api import detect
import ome_types
import ome_types.model as om
import typing

from ome_types.model.channel import *
from ome_types.model.detector import Detector, Type
from ome_types.model.dichroic import Dichroic
from ome_types.model.filter import Filter
from ome_types.model.filter_ref import FilterRef
from ome_types.model.filter_set import FilterSet
from ome_types.model.instrument import Instrument
from ome_types.model.microscope import Microscope
from ome_types.model.objective import Correction, Immersion, Objective
from ome_types.model.ome import OME
from ome_types.model.simple_types import PixelType, UnitsLength, UnitsTime
from ome_types.model.tiff_data import TiffData

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ome_types.model.transmittance_range import TransmittanceRange
from pydantic.utils import ClassAttribute


class MetadataEditor(QWidget):

    def __init__(self, parent: typing.Optional['QWidget'] = None):
        super().__init__(parent=parent)

        # self.setTitle('OME-XML Metadata')

        self.InitLayout()

    def InitLayout(self):

        self.main_layout = QVBoxLayout(self)
        self.qscroll = QScrollArea()
        self.main_layout.addWidget(self.qscroll)

        self.qscroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.qscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.qscroll.setWidgetResizable(True)

        self.TabView = QTabWidget()
        self.qscroll.setWidget(self.TabView)

        self.exp_tab = QWidget()
        self.exp_lay = QVBoxLayout()
        self.exp_tab.setLayout(self.exp_lay)
        self.image_tab = QWidget()
        self.image_lay = QVBoxLayout()
        self.image_tab.setLayout(self.image_lay)
        self.instruments_tab = QWidget()
        self.instruments_lay = QVBoxLayout()
        self.instruments_tab.setLayout(self.instruments_lay)

        self.TabView.addTab(self.exp_tab, 'Experiment')
        self.TabView.addTab(self.image_tab, 'Image')
        self.TabView.addTab(self.instruments_tab, 'Instruments')

        self.dValid = QDoubleValidator()
        self.iValid = QIntValidator()

        self.experiment = QLineEdit('Experiment_001')
        self.exp_lay.addWidget(QLabel('Experiment name:'))
        self.exp_lay.addWidget(self.experiment)

        self.exp_desc = QTextEdit('')
        self.exp_lay.addWidget(QLabel('Experiment description:'))
        self.exp_lay.addWidget(self.exp_desc)

        self.exp_email = QLineEdit()
        self.exp_fName = QLineEdit()
        self.exp_lName = QLineEdit()
        self.exp_institute = QLineEdit()

        self.exp_lay.addWidget(QLabel('Experimenter First name:'))
        self.exp_lay.addWidget(self.exp_fName)
        self.exp_lay.addWidget(QLabel('Experimenter Last name:'))
        self.exp_lay.addWidget(self.exp_lName)
        self.exp_lay.addWidget(QLabel('Experimenter e-mail:'))
        self.exp_lay.addWidget(self.exp_email)
        self.exp_lay.addWidget(QLabel('Experimenter Institute:'))
        self.exp_lay.addWidget(self.exp_institute)

        self.pixel_type = QComboBox()
        self.pixel_type.addItems(PixelType._member_names_)
        self.pixel_type.setCurrentText(PixelType.UINT16.name)

        self.px_size = QLineEdit('120')
        self.px_size.setValidator(self.dValid)
        self.py_size = QLineEdit('120')
        self.py_size.setValidator(self.dValid)
        self.p_unit = QComboBox()
        self.p_unit.addItems(UnitsLength._member_names_)
        self.p_unit.setCurrentText(UnitsLength.NANOMETER.name)

        self.image_lay.addWidget(QLabel('Pixel type:'))
        self.image_lay.addWidget(self.pixel_type)
        self.image_lay.addWidget(QLabel('Pixel x-size:'))
        self.image_lay.addWidget(self.px_size)
        self.image_lay.addWidget(QLabel('Pixel y-size:'))
        self.image_lay.addWidget(self.py_size)
        self.image_lay.addWidget(QLabel('Unit:'))
        self.image_lay.addWidget(self.p_unit)

        self.channel_name = QLineEdit('CAM_1')
        self.fluor_name = QLineEdit()

        self.image_lay.addWidget(QLabel('Channel name:'))
        self.image_lay.addWidget(self.channel_name)
        self.image_lay.addWidget(QLabel('Fluorophore:'))
        self.image_lay.addWidget(self.fluor_name)

        self.exposure = QLineEdit('100.0')
        self.exposure.setValidator(self.dValid)
        self.exposure_unit = QComboBox()
        self.exposure_unit.addItems(UnitsTime._member_names_)
        self.exposure_unit.setCurrentText(UnitsTime.MILLISECOND.name)
        self.image_lay.addWidget(QLabel('Exposure time:'))
        self.image_lay.addWidget(self.exposure)
        self.image_lay.addWidget(QLabel('Exposure time unit:'))
        self.image_lay.addWidget(self.exposure_unit)

        self.acq_mode = QComboBox()
        self.acq_mode.addItems(AcquisitionMode._member_names_)
        self.acq_mode.setCurrentText(AcquisitionMode.TIRF.name)

        self.image_lay.addWidget(QLabel('Acquisition mode:'))
        self.image_lay.addWidget(self.acq_mode)

        self.ill_type = QComboBox()
        self.ill_type.addItems(IlluminationType._member_names_)
        self.ill_type.setCurrentText(IlluminationType.OTHER.name)

        self.image_lay.addWidget(QLabel('Illumination type:'))
        self.image_lay.addWidget(self.ill_type)

        self.contrast = QComboBox()
        self.contrast.addItems(ContrastMethod._member_names_)
        self.contrast.setCurrentText(ContrastMethod.FLUORESCENCE.name)

        self.image_lay.addWidget(QLabel('Contrast method:'))
        self.image_lay.addWidget(self.contrast)

        self.emission = QLineEdit('670')
        self.emission.setValidator(self.dValid)
        self.excitation = QLineEdit('638')
        self.excitation.setValidator(self.dValid)
        self.wave_unit = QComboBox()
        self.wave_unit.addItems(UnitsLength._member_names_)
        self.wave_unit.setCurrentText(UnitsLength.NANOMETER.name)

        self.image_lay.addWidget(QLabel('Excitation wavelength:'))
        self.image_lay.addWidget(self.excitation)
        self.image_lay.addWidget(QLabel('Emission wavelength:'))
        self.image_lay.addWidget(self.emission)
        self.image_lay.addWidget(QLabel('Wavelength unit:'))
        self.image_lay.addWidget(self.wave_unit)

        self.micro_manufacturer = QLineEdit('VU/FTMC')
        self.micro_model = QLineEdit('Main scope')

        self.obj_manufacturer = QLineEdit('Nikon')
        self.obj_model = QLineEdit('CFI Apochromat TIRF 60XC Oil')
        self.obj_lens_na = QLineEdit('1.49')
        self.obj_lens_na.setValidator(self.dValid)
        self.obj_nom_mag = QLineEdit('60')
        self.obj_nom_mag.setValidator(self.dValid)
        self.obj_immersion = QComboBox()
        self.obj_immersion.addItems(Immersion._member_names_)
        self.obj_immersion.setCurrentText(Immersion.OIL.name)
        self.obj_corr = QComboBox()
        self.obj_corr.addItems(Correction._member_names_)
        self.obj_corr.setCurrentText(Correction.APO.name)

        self.instruments_lay.addWidget(QLabel('Microscope manufacturer:'))
        self.instruments_lay.addWidget(self.micro_manufacturer)
        self.instruments_lay.addWidget(QLabel('Microscope model:'))
        self.instruments_lay.addWidget(self.micro_model)
        self.instruments_lay.addWidget(QLabel('Objective manufacturer:'))
        self.instruments_lay.addWidget(self.obj_manufacturer)
        self.instruments_lay.addWidget(QLabel('Objective model:'))
        self.instruments_lay.addWidget(self.obj_model)
        self.instruments_lay.addWidget(QLabel('Objective NA:'))
        self.instruments_lay.addWidget(self.obj_lens_na)
        self.instruments_lay.addWidget(QLabel('Objective nominal Mag.:'))
        self.instruments_lay.addWidget(self.obj_nom_mag)
        self.instruments_lay.addWidget(QLabel('Objective immersion:'))
        self.instruments_lay.addWidget(self.obj_immersion)
        self.instruments_lay.addWidget(QLabel('Objective correction:'))
        self.instruments_lay.addWidget(self.obj_corr)

        self.det_manufacturer = QLineEdit('IDS')
        self.det_model = QLineEdit('ui-3060cp-rev-2')
        self.det_serial = QLineEdit('')
        self.det_type = QComboBox()
        self.det_type.addItems(Type._member_names_)
        self.det_type.setCurrentText(Type.CMOS.name)

        self.instruments_lay.addWidget(QLabel('Detector manufacturer:'))
        self.instruments_lay.addWidget(self.det_manufacturer)
        self.instruments_lay.addWidget(QLabel('Detector model:'))
        self.instruments_lay.addWidget(self.det_model)
        self.instruments_lay.addWidget(QLabel('Detector serial number:'))
        self.instruments_lay.addWidget(self.det_serial)
        self.instruments_lay.addWidget(QLabel('Detector type:'))
        self.instruments_lay.addWidget(self.det_type)

        self.dichroic_manufacturer = QLineEdit('')
        self.dichroic_model = QLineEdit('')

        self.instruments_lay.addWidget(QLabel('Dichroic manufacturer:'))
        self.instruments_lay.addWidget(self.dichroic_manufacturer)
        self.instruments_lay.addWidget(QLabel('Dichroic model:'))
        self.instruments_lay.addWidget(self.dichroic_model)

        self.exfilter_model = QLineEdit('')
        self.emfilter_model = QLineEdit('')

        self.instruments_lay.addWidget(QLabel('Excitation Filters:'))
        self.instruments_lay.addWidget(self.exfilter_model)
        self.instruments_lay.addWidget(QLabel('Emission Filters:'))
        self.instruments_lay.addWidget(self.emfilter_model)

        self.main_layout.addWidget(QPushButton(
            'Save as',
            clicked=lambda: self.saveAs()))

        self.main_layout.addWidget(QPushButton(
            'Import',
            clicked=lambda: self.loadXML()))

        self.setLayout(self.main_layout)

    def saveAs(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save metadata", filter="OME-XML Files (*.ome.xml);;")

        if len(filename) > 0:
            ome_obj = self.gen_OME_XML(1, 512, 512)
            with open(filename, 'w', encoding='utf8') as f:
                f.write(ome_obj.to_xml())

    def loadXML(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load metadata", filter="OME-XML Files (*.ome.xml);;")

        if len(filename) > 0:
            xml = ''
            with open(filename, 'r') as f:
                xml = f.read()
            self.pop_OME_XML(OME.from_xml(xml))

    def gen_OME_XML(self, frames: int, width: int, height: int):
        ome_obj = om.OME(creator='microEye Python Package')

        experimenter = om.Experimenter()
        experimenter.first_name = self.exp_fName.text()
        experimenter.last_name = self.exp_lName.text()
        experimenter.email = self.exp_email.text()
        experimenter.institution = self.exp_institute.text()
        ome_obj.experimenters.append(experimenter)

        micro = Microscope()
        micro.manufacturer = self.micro_manufacturer.text()
        micro.model = self.micro_model.text()

        objective = Objective()
        objective.manufacturer = self.obj_manufacturer.text()
        objective.model = self.obj_model.text()
        objective.lens_na = float(self.obj_lens_na.text())
        objective.nominal_magnification = float(self.obj_nom_mag.text())
        objective.immersion = Immersion._member_map_[
            self.obj_immersion.currentText()]
        objective.correction = Correction._member_map_[
            self.obj_corr.currentText()]

        detector = Detector()
        detector.manufacturer = self.det_manufacturer.text()
        detector.model = self.det_model.text()
        detector.serial_number = self.det_serial.text()
        detector.type = Type._member_map_[
            self.det_type.currentText()]

        dichroic = Dichroic()
        dichroic.manufacturer = self.dichroic_manufacturer.text()
        dichroic.model = self.dichroic_model.text()

        exFilters = Filter()
        exFilters.model = self.exfilter_model.text()
        emFilters = Filter()
        emFilters.model = self.emfilter_model.text()

        instrument = Instrument()
        instrument.microscope = micro
        instrument.objectives.append(objective)
        instrument.detectors.append(detector)
        instrument.dichroics.append(dichroic)
        instrument.filters.append(exFilters)
        instrument.filters.append(emFilters)

        ome_obj.instruments.append(instrument)

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
            type=PixelType._member_map_[
                self.pixel_type.currentText()],
            dimension_order='XYZCT',
            # metadata_only=True,
            physical_size_x=float(self.px_size.text()),
            physical_size_x_unit=UnitsLength._member_map_[
                self.p_unit.currentText()],
            physical_size_y=float(self.py_size.text()),
            physical_size_y_unit=UnitsLength._member_map_[
                self.p_unit.currentText()],
            time_increment=float(self.exposure.text()),
            time_increment_unit=UnitsTime._member_map_[
                self.exposure_unit.currentText()],
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

        # with open('config.xml', 'w', encoding='utf8') as f:
        #     f.write(ome_obj.to_xml())

        return ome_obj

    def pop_OME_XML(self, ome_obj: OME):
        if ome_obj.images.__len__() > 0:
            img = ome_obj.images[0]
            self.experiment.setText(img.name)
            self.exp_desc.setText(img.description)
            if img.pixels is not None:
                pixels = img.pixels
                self.px_size.setText(pixels.physical_size_x.__str__())
                self.py_size.setText(pixels.physical_size_y.__str__())
                self.p_unit.setCurrentText(pixels.physical_size_x_unit.name)
                self.exposure.setText(pixels.time_increment.__str__())
                self.exposure_unit.setCurrentText(
                    pixels.time_increment_unit.name)
                self.pixel_type.setCurrentText(pixels.type.name)
                if pixels.channels.__len__() > 0:
                    channel = pixels.channels[0]
                    self.channel_name.setText(channel.name)
                    self.fluor_name.setText(channel.fluor)
                    self.acq_mode.setCurrentText(channel.acquisition_mode.name)
                    self.ill_type.setCurrentText(
                        channel.illumination_type.name)
                    self.contrast.setCurrentText(channel.contrast_method.name)
                    self.excitation.setText(
                        channel.excitation_wavelength.__str__())
                    self.emission.setText(
                        channel.emission_wavelength.__str__())
                    self.wave_unit.setCurrentText(
                        channel.excitation_wavelength_unit.name)
                    self.wave_unit.setCurrentText(
                        channel.emission_wavelength_unit.name)
            if ome_obj.experimenters.__len__() > 0:
                exper = ome_obj.experimenters[0]
                self.exp_fName.setText(exper.first_name)
                self.exp_lName.setText(exper.last_name)
                self.exp_email.setText(exper.email)
                self.exp_institute.setText(exper.institution)
            if ome_obj.instruments.__len__() > 0:
                inst = ome_obj.instruments[0]
                if inst.microscope is not None:
                    micro = inst.microscope
                    self.micro_manufacturer.setText(micro.manufacturer)
                    self.micro_model.setText(micro.model)
                if inst.objectives.__len__() > 0:
                    objective = inst.objectives[0]
                    self.obj_manufacturer.setText(objective.manufacturer)
                    self.obj_model.setText(objective.model)
                    self.obj_lens_na.setText(objective.lens_na.__str__())
                    self.obj_nom_mag.setText(
                        objective.nominal_magnification.__str__())
                    self.obj_immersion.setCurrentText(objective.immersion.name)
                    self.obj_corr.setCurrentText(objective.correction.name)
                if inst.detectors.__len__() > 0:
                    detector = inst.detectors[0]
                    self.det_manufacturer.setText(detector.manufacturer)
                    self.det_model.setText(detector.model)
                    self.det_serial.setText(detector.serial_number)
                    self.det_type.setCurrentText(detector.type.name)
                if inst.dichroics.__len__() > 0:
                    dichroic = inst.dichroics[0]
                    self.dichroic_manufacturer.setText(dichroic.manufacturer)
                    self.dichroic_model.setText(dichroic.model)
                if inst.filters.__len__() > 1:
                    exFilters = inst.filters[0]
                    self.exfilter_model.setText(exFilters.model)
                    emFilters = inst.filters[1]
                    self.emfilter_model.setText(emFilters.model)
