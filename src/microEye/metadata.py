
import ome_types.model as om
import typing

from ome_types.model.channel import *
from ome_types.model.detector import Detector, Type
from ome_types.model.dichroic import Dichroic
from ome_types.model.filter import Filter
from ome_types.model.instrument import Instrument
from ome_types.model.microscope import Microscope
from ome_types.model.objective import Correction, Immersion, Objective
from ome_types.model.ome import OME
from ome_types.model.simple_types import PixelType, UnitsLength, UnitsTime

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class MetadataEditor(QWidget):

    def __init__(self, parent: typing.Optional['QWidget'] = None):
        super().__init__(parent=parent)

        # self.setTitle('OME-XML Metadata')

        self.InitLayout()

    def InitLayout(self):

        self.main_layout = QVBoxLayout(self)
        # self.qscroll = QScrollArea()

        # self.qscroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.qscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.qscroll.setWidgetResizable(True)

        self.TabView = QTabWidget()
        self.main_layout.addWidget(self.TabView)
        # self.qscroll.setWidget(self.TabView)

        self.exp_tab = QWidget()
        self.exp_lay = QFormLayout()
        self.exp_tab.setLayout(self.exp_lay)
        self.image_tab = QWidget()
        self.image_lay = QFormLayout()
        self.image_tab.setLayout(self.image_lay)
        self.instruments_tab = QWidget()
        self.instruments_lay = QFormLayout()
        self.instruments_tab.setLayout(self.instruments_lay)

        self.TabView.addTab(self.exp_tab, 'Experiment')
        self.TabView.addTab(self.image_tab, 'Image')
        self.TabView.addTab(self.instruments_tab, 'Instruments')

        self.dValid = QDoubleValidator()
        self.iValid = QIntValidator()

        self.experiment = QLineEdit('Experiment_001')
        self.exp_lay.addRow(
            QLabel('Experiment name:'),
            self.experiment)

        self.exp_desc = QTextEdit('')
        self.exp_lay.addRow(
            QLabel('Experiment description:'))
        self.exp_lay.addRow(self.exp_desc)

        self.exp_email = QLineEdit()
        self.exp_fName = QLineEdit()
        self.exp_lName = QLineEdit()
        self.exp_institute = QLineEdit()

        self.exp_lay.addRow(
            QLabel('Experimenter First name:'),
            self.exp_fName)
        self.exp_lay.addRow(
            QLabel('Experimenter Last name:'),
            self.exp_lName)
        self.exp_lay.addRow(
            QLabel('Experimenter e-mail:'),
            self.exp_email)
        self.exp_lay.addRow(
            QLabel('Experimenter Institute:'),
            self.exp_institute)

        self.pixel_type = QComboBox()
        self.pixel_type.addItems(PixelType._member_names_)
        self.pixel_type.setCurrentText(PixelType.UINT16.name)

        self.px_size = QDoubleSpinBox()
        self.px_size.setMinimum(0)
        self.px_size.setMaximum(10000)
        self.px_size.setValue(117.5)
        self.py_size = QDoubleSpinBox()
        self.py_size.setMinimum(0)
        self.py_size.setMaximum(10000)
        self.py_size.setValue(117.5)
        self.p_unit = QComboBox()
        self.p_unit.addItems(UnitsLength._member_names_)
        self.p_unit.setCurrentText(UnitsLength.NANOMETER.name)

        self.image_lay.addRow(
            QLabel('Pixel type:'),
            self.pixel_type)
        self.image_lay.addRow(
            QLabel('Pixel x-size:'),
            self.px_size)
        self.image_lay.addRow(
            QLabel('Pixel y-size:'),
            self.py_size)
        self.image_lay.addRow(
            QLabel('Unit:'),
            self.p_unit)

        self.channel_name = QLineEdit('CAM_1')
        self.fluor_name = QLineEdit()

        self.image_lay.addRow(
            QLabel('Channel name:'),
            self.channel_name)
        self.image_lay.addRow(
            QLabel('Fluorophore:'),
            self.fluor_name)

        self.exposure = QDoubleSpinBox()
        self.exposure.setMinimum(0)
        self.exposure.setMaximum(10000)
        self.exposure.setValue(100.0)
        self.exposure_unit = QComboBox()
        self.exposure_unit.addItems(UnitsTime._member_names_)
        self.exposure_unit.setCurrentText(UnitsTime.MILLISECOND.name)
        self.image_lay.addRow(
            QLabel('Exposure time:'),
            self.exposure)
        self.image_lay.addRow(
            QLabel('Exposure time unit:'),
            self.exposure_unit)

        self.acq_mode = QComboBox()
        self.acq_mode.addItems(AcquisitionMode._member_names_)
        self.acq_mode.setCurrentText(AcquisitionMode.TIRF.name)

        self.image_lay.addRow(
            QLabel('Acquisition mode:'),
            self.acq_mode)

        self.ill_type = QComboBox()
        self.ill_type.addItems(IlluminationType._member_names_)
        self.ill_type.setCurrentText(IlluminationType.OTHER.name)

        self.image_lay.addRow(
            QLabel('Illumination type:'),
            self.ill_type)

        self.contrast = QComboBox()
        self.contrast.addItems(ContrastMethod._member_names_)
        self.contrast.setCurrentText(ContrastMethod.FLUORESCENCE.name)

        self.image_lay.addRow(
            QLabel('Contrast method:'),
            self.contrast)

        self.emission = QDoubleSpinBox()
        self.emission.setMinimum(0)
        self.emission.setMaximum(10000)
        self.emission.setValue(670)
        self.excitation = QDoubleSpinBox()
        self.excitation.setMinimum(0)
        self.excitation.setMaximum(10000)
        self.excitation.setValue(638)
        self.wave_unit = QComboBox()
        self.wave_unit.addItems(UnitsLength._member_names_)
        self.wave_unit.setCurrentText(UnitsLength.NANOMETER.name)

        self.image_lay.addRow(
            QLabel('Excitation wavelength:'),
            self.excitation)
        self.image_lay.addRow(
            QLabel('Emission wavelength:'),
            self.emission)
        self.image_lay.addRow(
            QLabel('Wavelength unit:'),
            self.wave_unit)

        self.micro_manufacturer = QLineEdit('VU/FTMC')
        self.micro_model = QLineEdit('Main scope')

        self.obj_manufacturer = QLineEdit('Nikon')
        self.obj_model = QLineEdit('CFI Apochromat TIRF 60XC Oil')
        self.obj_lens_na = QDoubleSpinBox()
        self.obj_lens_na.setMinimum(0)
        self.obj_lens_na.setMaximum(2)
        self.obj_lens_na.setValue(1.49)
        self.obj_nom_mag = QDoubleSpinBox()
        self.obj_nom_mag.setMinimum(0)
        self.obj_nom_mag.setMaximum(1000)
        self.obj_nom_mag.setValue(60)
        self.obj_immersion = QComboBox()
        self.obj_immersion.addItems(Immersion._member_names_)
        self.obj_immersion.setCurrentText(Immersion.OIL.name)
        self.obj_corr = QComboBox()
        self.obj_corr.addItems(Correction._member_names_)
        self.obj_corr.setCurrentText(Correction.APO.name)

        self.instruments_lay.addRow(
            QLabel('Microscope manufacturer:'),
            self.micro_manufacturer)
        self.instruments_lay.addRow(
            QLabel('Microscope model:'),
            self.micro_model)
        self.instruments_lay.addRow(
            QLabel('Objective manufacturer:'),
            self.obj_manufacturer)
        self.instruments_lay.addRow(
            QLabel('Objective model:'),
            self.obj_model)
        self.instruments_lay.addRow(
            QLabel('Objective NA:'),
            self.obj_lens_na)
        self.instruments_lay.addRow(
            QLabel('Objective nominal Mag.:'),
            self.obj_nom_mag)
        self.instruments_lay.addRow(
            QLabel('Objective immersion:'),
            self.obj_immersion)
        self.instruments_lay.addRow(
            QLabel('Objective correction:'),
            self.obj_corr)

        self.det_manufacturer = QLineEdit('IDS')
        self.det_model = QLineEdit('ui-3060cp-rev-2')
        self.det_serial = QLineEdit('')
        self.det_type = QComboBox()
        self.det_type.addItems(Type._member_names_)
        self.det_type.setCurrentText(Type.CMOS.name)

        self.instruments_lay.addRow(
            QLabel('Detector manufacturer:'),
            self.det_manufacturer)
        self.instruments_lay.addRow(
            QLabel('Detector model:'),
            self.det_model)
        self.instruments_lay.addRow(
            QLabel('Detector serial number:'),
            self.det_serial)
        self.instruments_lay.addRow(
            QLabel('Detector type:'),
            self.det_type)

        self.dichroic_manufacturer = QLineEdit('')
        self.dichroic_model = QLineEdit('')

        self.instruments_lay.addRow(
            QLabel('Dichroic manufacturer:'),
            self.dichroic_manufacturer)
        self.instruments_lay.addRow(
            QLabel('Dichroic model:'),
            self.dichroic_model)

        self.exfilter_model = QLineEdit('')
        self.emfilter_model = QLineEdit('')

        self.instruments_lay.addRow(
            QLabel('Excitation Filters:'),
            self.exfilter_model)
        self.instruments_lay.addRow(
            QLabel('Emission Filters:'),
            self.emfilter_model)

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
        objective.lens_na = self.obj_lens_na.value()
        objective.nominal_magnification = self.obj_nom_mag.value()
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
            exposure_time=self.exposure.value(),
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
        channel.excitation_wavelength = self.excitation.value()
        channel.emission_wavelength = self.emission.value()
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
            physical_size_x=self.px_size.value(),
            physical_size_x_unit=UnitsLength._member_map_[
                self.p_unit.currentText()],
            physical_size_y=self.py_size.value(),
            physical_size_y_unit=UnitsLength._member_map_[
                self.p_unit.currentText()],
            time_increment=self.exposure.value(),
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

    def gen_OME_XML_short(self, frames: int, width: int, height: int):
        ome_obj = om.OME(creator='microEye Python Package')

        experimenter = om.Experimenter()
        experimenter.first_name = self.exp_fName.text()
        experimenter.last_name = self.exp_lName.text()
        experimenter.email = self.exp_email.text()
        experimenter.institution = self.exp_institute.text()
        ome_obj.experimenters.append(experimenter)

        detector = Detector()
        detector.manufacturer = self.det_manufacturer.text()
        detector.model = self.det_model.text()
        detector.serial_number = self.det_serial.text()
        detector.type = Type._member_map_[
            self.det_type.currentText()]

        instrument = Instrument()
        instrument.detectors.append(detector)

        ome_obj.instruments.append(instrument)

        planes = [om.Plane(
            the_c=0,
            the_t=i,
            the_z=0,
            exposure_time=self.exposure.value(),
            exposure_time_unit=UnitsTime.MILLISECOND
        ) for i in range(frames)]

        channel = om.Channel()

        pixels = om.Pixels(
            size_c=1, size_t=frames,
            size_x=width,
            size_y=height, size_z=1,
            type=PixelType._member_map_[
                self.pixel_type.currentText()],
            dimension_order='XYZCT',
            # metadata_only=True,
            physical_size_x=self.px_size.value(),
            physical_size_x_unit=UnitsLength._member_map_[
                self.p_unit.currentText()],
            physical_size_y=self.py_size.value(),
            physical_size_y_unit=UnitsLength._member_map_[
                self.p_unit.currentText()],
            time_increment=self.exposure.value(),
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

        return ome_obj

    def pop_OME_XML(self, ome_obj: OME):
        if ome_obj.images.__len__() > 0:
            img = ome_obj.images[0]
            self.experiment.setText(img.name)
            self.exp_desc.setText(img.description)
            if img.pixels is not None:
                pixels = img.pixels
                if pixels.physical_size_x is not None:
                    self.px_size.setValue(
                        float(pixels.physical_size_x))
                    self.p_unit.setCurrentText(
                        pixels.physical_size_x_unit.name)
                if pixels.physical_size_y is not None:
                    self.py_size.setValue(
                        float(pixels.physical_size_y))
                if pixels.time_increment is not None:
                    self.exposure.setValue(pixels.time_increment)
                    self.exposure_unit.setCurrentText(
                        pixels.time_increment_unit.name)
                self.pixel_type.setCurrentText(pixels.type.name)
                if pixels.channels.__len__() > 0:
                    channel = pixels.channels[0]
                    self.channel_name.setText(channel.name)
                    self.fluor_name.setText(channel.fluor)
                    if channel.acquisition_mode:
                        self.acq_mode.setCurrentText(
                            channel.acquisition_mode.name)
                    if channel.illumination_type:
                        self.ill_type.setCurrentText(
                            channel.illumination_type.name)
                    if channel.contrast_method:
                        self.contrast.setCurrentText(
                            channel.contrast_method.name)
                    if channel.excitation_wavelength:
                        self.excitation.setValue(
                            float(channel.excitation_wavelength))
                        self.wave_unit.setCurrentText(
                            channel.excitation_wavelength_unit.name)
                    if channel.emission_wavelength:
                        self.emission.setValue(
                            float(channel.emission_wavelength))
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
                    self.obj_lens_na.setValue(objective.lens_na)
                    self.obj_nom_mag.setValue(
                        objective.nominal_magnification)
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
