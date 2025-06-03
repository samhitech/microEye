import typing

import ome_types.model as om
from ome_types.model import *
from ome_types.model.simple_types import PixelType, UnitsLength, UnitsTime

from microEye.qt import QApplication, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.gui_helper import (
    create_combo_box,
    create_labelled_double_spin_box,
    create_line_edit,
    create_text_edit,
)


class MetadataEditor(QtWidgets.QWidget):
    '''
    A PyQt-based GUI for editing OME-XML metadata.

    Attributes
    ----------
    experiment : QLineEdit
        Experiment name input field.
    exp_desc : QTextEdit
        Experiment description input field.
    exp_email : QLineEdit
        Experimenter's email input field.
    exp_fName : QLineEdit
        Experimenter's first name input field.
    exp_lName : QLineEdit
        Experimenter's last name input field.
    exp_institute : QLineEdit
        Experimenter's institute input field.
    pixel_type : QComboBox
        Pixel type selection dropdown.
    px_size : QDoubleSpinBox
        Pixel x-size input field.
    py_size : QDoubleSpinBox
        Pixel y-size input field.
    p_unit : QComboBox
        Unit selection dropdown for pixel size.
    channel_name : QLineEdit
        Channel name input field.
    fluor_name : QLineEdit
        Fluorophore input field.
    exposure : QDoubleSpinBox
        Exposure time input field.
    exposure_unit : QComboBox
        Exposure time unit selection dropdown.
    acq_mode : QComboBox
        Acquisition mode selection dropdown.
    ill_type : QComboBox
        Illumination type selection dropdown.
    contrast : QComboBox
        Contrast method selection dropdown.
    excitation : QDoubleSpinBox
        Excitation wavelength input field.
    emission : QDoubleSpinBox
        Emission wavelength input field.
    wave_unit : QComboBox
        Wavelength unit selection dropdown.
    micro_manufacturer : QLineEdit
        Microscope manufacturer input field.
    micro_model : QLineEdit
        Microscope model input field.
    obj_manufacturer : QLineEdit
        Objective manufacturer input field.
    obj_model : QLineEdit
        Objective model input field.
    obj_lens_na : QDoubleSpinBox
        Objective numerical aperture input field.
    obj_nom_mag : QDoubleSpinBox
        Objective nominal magnification input field.
    obj_immersion : QComboBox
        Objective immersion selection dropdown.
    obj_corr : QComboBox
        Objective correction selection dropdown.
    det_manufacturer : QLineEdit
        Detector manufacturer input field.
    det_model : QLineEdit
        Detector model input field.
    det_serial : QLineEdit
        Detector serial number input field.
    det_type : QComboBox
        Detector type selection dropdown.
    dichroic_manufacturer : QLineEdit
        Dichroic manufacturer input field.
    dichroic_model : QLineEdit
        Dichroic model input field.
    exfilter_model : QLineEdit
        Excitation filter model input field.
    emfilter_model : QLineEdit
        Emission filter model input field.
    tab_view : QTabWidget
        Tab view for organizing metadata sections.
    main_layout : QVBoxLayout
        Main layout of the widget.
    '''


    def __init__(self, parent: typing.Optional['QtWidgets.QWidget'] = None):
        '''
        Initializes the MetadataEditor widget.

        Parameters
        ----------
        parent : Optional[QWidget], optional
            The parent widget, by default None.
        '''
        super().__init__(parent=parent)
        self.setMinimumWidth(50)
        self.InitLayout()

    def InitLayout(self):
        '''
        Initializes the main layout of the widget.
        '''
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.tab_view = QtWidgets.QTabWidget()
        self.main_layout.addWidget(self.tab_view)

        self.create_tab('Experiment', self.exp_tab_setup)
        self.create_tab('Image', self.image_tab_setup)
        self.create_tab('Instruments', self.instruments_tab_setup)

        self.main_layout.addWidget(
            QtWidgets.QPushButton('Save as', clicked=self.save))
        self.main_layout.addWidget(
            QtWidgets.QPushButton('Import', clicked=self.load_xml))
        self.setLayout(self.main_layout)

    def create_tab(self, title, setup_func):
        '''
        Creates a tab in the tab view.

        Parameters
        ----------
        title : str
            Title of the tab.
        setup_func : callable
            Function for setting up the tab layout.
        '''
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()
        tab.setLayout(layout)
        self.tab_view.addTab(tab, title)
        setup_func(layout)

    def exp_tab_setup(self, layout):
        '''
        Sets up the Experiment tab.

        Parameters
        ----------
        layout : QFormLayout
            Layout for the Experiment tab.
        '''
        self.experiment = create_line_edit('Experiment name:', '001_Experiment', layout)
        self.exp_desc = create_text_edit('Experiment description:', '', layout)
        self.exp_email = create_line_edit('Experimenter e-mail:', '', layout)
        self.exp_fName = create_line_edit('Experimenter First name:', '', layout)
        self.exp_lName = create_line_edit('Experimenter Last name:', '', layout)
        self.exp_institute = create_line_edit('Experimenter Institute:', '', layout)

    def image_tab_setup(self, layout):
        '''
        Sets up the Image tab.

        Parameters
        ----------
        layout : QFormLayout
            Layout for the Image tab.
        '''
        self.pixel_type = create_combo_box(
            'Pixel type:', PixelType._member_names_,
            PixelType.UINT16.name, layout)
        self.px_size = create_labelled_double_spin_box(
            'Pixel x-size:', 0, 10000, 117.5, layout)
        self.py_size = create_labelled_double_spin_box(
            'Pixel y-size:', 0, 10000, 117.5, layout)
        self.p_unit = create_combo_box(
            'Unit:', UnitsLength._member_names_,
            UnitsLength.NANOMETER.name, layout)

        self.channel_name = create_line_edit('Channel name:', 'CAM_1', layout)
        self.fluor_name = create_line_edit('Fluorophore:', '', layout)
        self.exposure = create_labelled_double_spin_box(
            'Exposure time:', 0, 10000, 100.0, layout)
        self.exposure_unit = create_combo_box(
            'Exposure time unit:', UnitsTime._member_names_,
            UnitsTime.MILLISECOND.name, layout)
        self.acq_mode = create_combo_box(
            'Acquisition mode:', Channel_AcquisitionMode._member_names_,
            Channel_AcquisitionMode.TIRF.name, layout)
        self.ill_type = create_combo_box(
            'Illumination type:', Channel_IlluminationType._member_names_,
            Channel_IlluminationType.OTHER.name, layout)
        self.contrast = create_combo_box(
            'Contrast method:', Channel_ContrastMethod._member_names_,
            Channel_ContrastMethod.FLUORESCENCE.name, layout)
        self.excitation = create_labelled_double_spin_box(
            'Excitation wavelength:', 0, 10000, 638, layout)
        self.emission = create_labelled_double_spin_box(
            'Emission wavelength:', 0, 10000, 670, layout)
        self.wave_unit = create_combo_box(
            'Wavelength unit:', UnitsLength._member_names_,
            UnitsLength.NANOMETER.name, layout)

    def instruments_tab_setup(self, layout):
        '''
        Sets up the Instruments tab.

        Parameters
        ----------
        layout : QFormLayout
            Layout for the Instruments tab.
        '''
        self.micro_manufacturer = create_line_edit(
            'Microscope manufacturer:', 'VU/FTMC', layout)
        self.micro_model = create_line_edit(
            'Microscope model:', 'Main scope', layout)
        self.obj_manufacturer = create_line_edit(
            'Objective manufacturer:', 'Nikon', layout)
        self.obj_model = create_line_edit(
            'Objective model:', 'CFI Apochromat TIRF 60XC Oil', layout)
        self.obj_lens_na = create_labelled_double_spin_box(
            'Objective NA:', 0, 2, 1.49, layout)
        self.obj_nom_mag = create_labelled_double_spin_box(
            'Objective nominal Mag.:', 0, 1000, 60, layout)
        self.obj_immersion = create_combo_box(
            'Objective immersion:', Objective_Immersion._member_names_,
            Objective_Immersion.OIL.name, layout)
        self.obj_corr = create_combo_box(
            'Objective correction:', Objective_Correction._member_names_,
            Objective_Correction.APO.name, layout)
        self.det_manufacturer = create_line_edit(
            'Detector manufacturer:', 'Allied Vision', layout)
        self.det_model = create_line_edit(
            'Detector model:', 'U-511m', layout)
        self.det_serial = create_line_edit(
            'Detector serial number:', '', layout)
        self.det_type = create_combo_box(
            'Detector type:', Detector_Type._member_names_,
            Detector_Type.CMOS.name, layout)
        self.dichroic_manufacturer = create_line_edit(
            'Dichroic manufacturer:', '', layout)
        self.dichroic_model = create_line_edit(
            'Dichroic model:', '', layout)
        self.exfilter_model = create_line_edit(
            'Excitation Filters:', '', layout)
        self.emfilter_model = create_line_edit(
            'Emission Filters:', '', layout)

    def save(self):
        '''
        Saves metadata to an OME-XML file.
        '''
        try:
            filename, _ = getSaveFileName(
                self, 'Save metadata', filter='OME-XML Files (*.ome.xml);;')
            if filename:
                ome_obj = self.gen_OME_XML(1, 512, 512)
                with open(filename, 'w', encoding='utf8') as f:
                    f.write(ome_obj.to_xml())
        except Exception as e:
            print(f'Error saving file: {e}')

    def load_xml(self):
        '''
        Loads metadata from an OME-XML file.
        '''
        try:
            filename, _ = getOpenFileName(
                self, 'Load metadata', filter='OME-XML Files (*.ome.xml);;')
            if filename:
                xml = ''
                with open(filename) as f:
                    xml = f.read()
                self.pop_OME_XML(OME.from_xml(xml))
        except Exception as e:
            print(f'Error loading file: {e}')


    def gen_OME_XML(
            self, frames: int, width: int, height: int,
            channels: int = 1, z_planes: int = 1,
            dimension_order: Pixels_DimensionOrder =Pixels_DimensionOrder.XYZCT):
        '''
        Generates OME-XML metadata.

        Parameters
        ----------
        frames : int
            Number of frames.
        width : int
            Image width.
        height : int
            Image height.
        channels : int, optional
            Number of channels, by default 1.
        z_planes : int, optional
            Number of z-planes, by default 1.
        dimension_order : Pixels_DimensionOrder, optional
            Dimension order, by default Pixels_DimensionOrder.XYZCT.

        Returns
        -------
        om.OME
            Generated OME-XML object.
        '''
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
        objective.immersion = Objective_Immersion._member_map_[
            self.obj_immersion.currentText()]
        objective.correction = Objective_Correction._member_map_[
            self.obj_corr.currentText()]

        detector = Detector()
        detector.manufacturer = self.det_manufacturer.text()
        detector.model = self.det_model.text()
        detector.serial_number = self.det_serial.text()
        detector.type = Detector_Type._member_map_[
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
        channel.acquisition_mode = Channel_AcquisitionMode._member_map_[
            self.acq_mode.currentText()]
        channel.illumination_type = Channel_IlluminationType._member_map_[
            self.ill_type.currentText()]
        channel.contrast_method = Channel_ContrastMethod._member_map_[
            self.contrast.currentText()]
        channel.excitation_wavelength = self.excitation.value()
        channel.emission_wavelength = self.emission.value()
        channel.excitation_wavelength_unit = UnitsLength._member_map_[
            self.wave_unit.currentText()]
        channel.emission_wavelength_unit = UnitsLength._member_map_[
            self.wave_unit.currentText()]
        channel.samples_per_pixel = 1

        pixels = om.Pixels(
            size_c=channels, size_t=frames,
            size_x=width,
            size_y=height,
            size_z=z_planes,
            type=PixelType._member_map_[
                self.pixel_type.currentText()],
            dimension_order=dimension_order,
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
        ome_obj = OME.model_validate(ome_obj)

        # with open('config.xml', 'w', encoding='utf8') as f:
        #     f.write(ome_obj.to_xml())

        return ome_obj

    def gen_OME_XML_short(
            self, frames: int, width: int, height: int,
            channels: int = 1, z_planes: int = 1,
            dimension_order: Pixels_DimensionOrder =Pixels_DimensionOrder.XYZCT):
        '''
        Generates short version of OME-XML metadata.

        Parameters
        ----------
        frames : int
            Number of frames.
        width : int
            Image width.
        height : int
            Image height.
        channels : int, optional
            Number of channels, by default 1.
        z_planes : int, optional
            Number of z-planes, by default 1.
        dimension_order : Pixels_DimensionOrder, optional
            Dimension order, by default Pixels_DimensionOrder.XYZCT.

        Returns
        -------
        om.OME
            Generated OME-XML object.
        '''
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
        detector.type = Detector_Type._member_map_[
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
            size_c=channels, size_t=frames,
            size_x=width,
            size_y=height, size_z=z_planes,
            type=PixelType._member_map_[
                self.pixel_type.currentText()],
            dimension_order=dimension_order,
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
        ome_obj = OME.model_validate(ome_obj)

        return ome_obj

    def pop_OME_XML(self, ome_obj: OME):
        '''
        Populate the widget with OME-XML metadata.

        Parameters
        ----------
        ome_obj : om.OME
            The OME-XML object to be loaded.
        '''
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MetadataEditor()
    window.show()
    sys.exit(app.exec())
