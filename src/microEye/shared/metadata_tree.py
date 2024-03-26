import json
from enum import Enum
from typing import Any, Optional, Union

import ome_types.model as om
from ome_types.model import *
from ome_types.model.simple_types import PixelType, UnitsLength, UnitsTime
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import ActionParameter, GroupParameter

from .parameter_tree import Tree


class MetaParams(Enum):
    '''
    Enum class defining metadata parameters.
    '''
    EXPERIMENT_NAME = 'Experiment.Name'
    EXP_DESC = 'Experiment.Description'
    EXP_EMAIL = 'Experiment.Email'
    EXP_FNAME = 'Experiment.First Name'
    EXP_LNAME = 'Experiment.Last Name'
    EXP_INSTITUTE = 'Experiment.Institute'
    PIXEL_TYPE = 'Image.Pixel Type'
    PX_SIZE = 'Image.Pixel X-Size'
    PY_SIZE = 'Image.Pixel Y-Size'
    P_UNIT = 'Image.Pixel Unit'
    CHANNEL_NAME = 'Image.Channel Name'
    FLUOR_NAME = 'Image.Fluorophore'
    EXPOSURE = 'Image.Exposure Time'
    EXPOSURE_UNIT = 'Image.Exposure Time Unit'
    ACQ_MODE = 'Image.Acquisition Mode'
    ILL_TYPE = 'Image.Illumination Type'
    CONTRAST = 'Image.Contrast Method'
    EXCITATION = 'Image.Excitation Wavelength'
    EMISSION = 'Image.Emission Wavelength'
    WAVE_UNIT = 'Image.Wavelength Unit'
    MICRO_MANUFACTURER = 'Instruments.Microscope Manufacturer'
    MICRO_MODEL = 'Instruments.Microscope Model'
    OBJ_MANUFACTURER = 'Instruments.Objective Manufacturer'
    OBJ_MODEL = 'Instruments.Objective Model'
    OBJ_LENS_NA = 'Instruments.Objective NA'
    OBJ_NOM_MAG = 'Instruments.Objective Nominal Magnification'
    OBJ_IMMERSION = 'Instruments.Objective Immersion'
    OBJ_CORR = 'Instruments.Objective Correction'
    DET_MANUFACTURER = 'Instruments.Detector Manufacturer'
    DET_MODEL = 'Instruments.Detector Model'
    DET_SERIAL = 'Instruments.Detector Serial Number'
    DET_TYPE = 'Instruments.Detector Type'
    DICHROIC_MANUFACTURER = 'Instruments.Dichroic Manufacturer'
    DICHROIC_MODEL = 'Instruments.Dichroic'
    EXFILTER_MODEL = 'Instruments.Excitation Filter'
    EMFILTER_MODEL = 'Instruments.Emission Filter'
    DICHROIC_MODEL_LIST = 'Instruments.Dichroics'
    EXFILTER_MODEL_LIST = 'Instruments.Excitation Filters'
    EMFILTER_MODEL_LIST = 'Instruments.Emission Filters'
    DICHROIC_MODEL_BTN = 'Instruments.Add Dichroic'
    EXFILTER_MODEL_BTN = 'Instruments.Add Excitation Filters'
    EMFILTER_MODEL_BTN = 'Instruments.Add Emission Filters'
    EXPORT_STATE = 'Actions.Export State'
    IMPORT_STATE = 'Actions.Import State'
    EXPORT_XML = 'Actions.Export XML'
    IMPORT_XML = 'Actions.Import XML'

    def __str__(self):
        '''
        Return the last part of the enum value (Param name).
        '''
        return self.value.split('.')[-1]

    def get_path(self):
        '''
        Return the full parameter path.
        '''
        return self.value.split('.')

class MetadataEditorTree(Tree):
    '''
    Tree widget for editing metadata parameters.

    Attributes
    ----------
    paramsChanged : pyqtSignal
        Signal for parameter changed event.

    DICHROIC_SUGGESTIONS : list
        List of dichroic suggestions, adjust to fit your setup.
    EMISSION_FILTERS : list
        List of emission filter suggestions, adjust to fit your setup.
    EXCITATION_FILTERS : list
        List of excitation filter suggestions, adjust to fit your setup.
    '''

    DICHROIC_SUGGESTIONS = [
        'EM 550 Longpass',
        'EM 640 Longpass',
        'EX 405/488/532/640 2mm MultiBand',
        'EX 405/488/561/635 3mm MultiBand',
        'EX 488/640 2mm MultiBand',
        'EX 405/514/647 2mm MultiBand',
        'EX 405/514/635 1mm MultiBand',
        'Other/None',
    ]
    '''List of dichroic suggestions, adjust to fit your setup.

    Suggestions for dichroic filters that can be used.
    '''

    EMISSION_FILTERS = [
        '697/75 Bandpass (1)',
        '692/40 Bandpass (2)',
        '630/69 Bandpass (9)',
        '591.5/43 Bandpass (3)',
        '575/35 Bandpass (4)',
        '550 Longpass (5)',
        '540/50 Bandpass (6)',
        '525/45 Bandpass (7)',
        '405/488/532/642 MultiBand (8)',
        'Other/None',
    ]
    '''List of emission filter suggestions, adjust to fit your setup.

    Suggestions for emission filters that can be used.
    '''

    EXCITATION_FILTERS = [
        '405/488/532/638 MultiBand',
        '640/10 BandPass',
        'Other/None',
    ]
    '''List of excitation filter suggestions, adjust to fit your setup.

    Suggestions for excitation filters that can be used.
    '''

    def __init__(self, parent: Optional['QWidget'] = None):
        '''
        Initialize the MetadataEditorTree.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None.
        '''
        super().__init__(parent=parent)

    def create_parameters(self):
        '''
        Create the parameter tree structure.
        '''
        params = [
            {'name': 'Experiment', 'type': 'group', 'children': [
                {'name': str(MetaParams.EXPERIMENT_NAME),
                 'type': 'str', 'value': 'Experiment_001'},
                {'name': str(MetaParams.EXP_DESC), 'type': 'text', 'value': ''},
                {'name': str(MetaParams.EXP_EMAIL), 'type': 'str', 'value': ''},
                {'name': str(MetaParams.EXP_FNAME), 'type': 'str', 'value': ''},
                {'name': str(MetaParams.EXP_LNAME), 'type': 'str', 'value': ''},
                {'name': str(MetaParams.EXP_INSTITUTE), 'type': 'str', 'value': ''},
            ]},
            {'name': 'Image', 'type': 'group', 'children': [
                {'name': str(MetaParams.PIXEL_TYPE), 'type': 'list',
                 'values': PixelType._member_names_, 'value': PixelType.UINT16.name},
                {'name': str(MetaParams.PX_SIZE), 'type': 'float',
                 'value': 117.5, 'limits': [0.0, 10000.0], 'decimals': 5},
                {'name': str(MetaParams.PY_SIZE), 'type': 'float',
                 'value': 117.5, 'limits': [0.0, 10000.0], 'decimals': 5},
                {'name': str(MetaParams.P_UNIT), 'type': 'list',
                 'values': UnitsLength._member_names_,
                 'value': UnitsLength.NANOMETER.name},
                {'name': str(MetaParams.CHANNEL_NAME), 'type': 'str', 'value': 'CAM_1'},
                {'name': str(MetaParams.FLUOR_NAME), 'type': 'str', 'value': ''},
                {'name': str(MetaParams.EXPOSURE), 'type': 'float',
                 'value': 100.0, 'limits': [0.0, 10000.0], 'decimals': 5},
                {'name': str(MetaParams.EXPOSURE_UNIT), 'type': 'list',
                 'values': UnitsTime._member_names_,
                 'value': UnitsTime.MILLISECOND.name},
                {'name': str(MetaParams.ACQ_MODE), 'type': 'list',
                 'values': Channel_AcquisitionMode._member_names_,
                 'value': Channel_AcquisitionMode.SINGLE_MOLECULE_IMAGING.name},
                {'name': str(MetaParams.ILL_TYPE), 'type': 'list',
                 'values': Channel_IlluminationType._member_names_,
                 'value': Channel_IlluminationType.OTHER.name},
                {'name': str(MetaParams.CONTRAST), 'type': 'list',
                 'values': Channel_ContrastMethod._member_names_,
                 'value': Channel_ContrastMethod.FLUORESCENCE.name},
                {'name': str(MetaParams.EXCITATION), 'type': 'float',
                 'value': 638, 'limits': [0.0, 10000.0], 'decimals': 5},
                {'name': str(MetaParams.EMISSION), 'type': 'float',
                 'value': 670, 'limits': [0.0, 10000.0], 'decimals': 5},
                {'name': str(MetaParams.WAVE_UNIT), 'type': 'list',
                 'values': UnitsLength._member_names_,
                 'value': UnitsLength.NANOMETER.name},
            ]},
            {'name': 'Instruments', 'type': 'group', 'children': [
                {'name': str(MetaParams.MICRO_MANUFACTURER),
                 'type': 'str', 'value': 'VU/FTMC'},
                {'name': str(MetaParams.MICRO_MODEL),
                 'type': 'str', 'value': 'Main scope'},
                {'name': str(MetaParams.OBJ_MANUFACTURER),
                 'type': 'str', 'value': 'Nikon'},
                {'name': str(MetaParams.OBJ_MODEL), 'type': 'str',
                 'value': 'CFI Apochromat TIRF 60XC Oil'},
                {'name': str(MetaParams.OBJ_LENS_NA), 'type': 'float',
                 'value': 1.49, 'limits': [0.0, 2.0], 'step': 0.1, 'decimals': 5},
                {'name': str(MetaParams.OBJ_NOM_MAG), 'type': 'float',
                 'value': 60.0, 'limits': [0.0, 1000.0], 'decimals': 5},
                {'name': str(MetaParams.OBJ_IMMERSION), 'type': 'list',
                 'values': Objective_Immersion._member_names_,
                 'value': Objective_Immersion.OIL.name},
                {'name': str(MetaParams.OBJ_CORR), 'type': 'list',
                 'values' : Objective_Correction._member_names_,
                 'value': Objective_Correction.PLAN_APO.name},
                {'name': str(MetaParams.DET_MANUFACTURER), 'type': 'str',
                 'value': 'Allied Vision'},
                {'name': str(MetaParams.DET_MODEL), 'type': 'str', 'value': 'U-511m'},
                {'name': str(MetaParams.DET_SERIAL), 'type': 'str', 'value': ''},
                {'name': str(MetaParams.DET_TYPE), 'type': 'list',
                 'values': Detector_Type._member_names_,
                 'value': Detector_Type.CMOS.name},
                {'name': str(MetaParams.DICHROIC_MANUFACTURER),
                 'type': 'str', 'value': ''},
                {'name': str(MetaParams.DICHROIC_MODEL), 'type': 'list', 'value': '',
                 'values': self.DICHROIC_SUGGESTIONS},
                {'name': str(MetaParams.DICHROIC_MODEL_BTN), 'type': 'action'},
                {'name': str(MetaParams.DICHROIC_MODEL_LIST), 'type': 'group',
                 'children': []},
                {'name': str(MetaParams.EXFILTER_MODEL), 'type': 'list', 'value': '',
                 'values': self.EXCITATION_FILTERS},
                {'name': str(MetaParams.EXFILTER_MODEL_BTN), 'type': 'action'},
                {'name': str(MetaParams.EXFILTER_MODEL_LIST), 'type': 'group',
                 'children': []},
                {'name': str(MetaParams.EMFILTER_MODEL), 'type': 'list', 'value': '',
                 'values': self.EMISSION_FILTERS},
                {'name': str(MetaParams.EMFILTER_MODEL_BTN), 'type': 'action'},
                {'name': str(MetaParams.EMFILTER_MODEL_LIST), 'type': 'group',
                 'children': []},
            ]},
            {'name': 'Actions', 'type': 'group', 'children': [
                {'name': str(MetaParams.EXPORT_STATE), 'type': 'action'},
                {'name': str(MetaParams.IMPORT_STATE), 'type': 'action'},
                {'name': str(MetaParams.EXPORT_XML), 'type': 'action'},
                {'name': str(MetaParams.IMPORT_XML), 'type': 'action'},
            ]},
        ]

        self.param_tree = Parameter.create(name='root', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

        self.get_param(
            MetaParams.DICHROIC_MODEL_BTN).sigActivated.connect(
                lambda: self.add_param_child(
                    MetaParams.DICHROIC_MODEL_LIST,
                    MetaParams.DICHROIC_MODEL))
        self.get_param(
            MetaParams.EXFILTER_MODEL_BTN).sigActivated.connect(
                lambda: self.add_param_child(
                    MetaParams.EXFILTER_MODEL_LIST,
                    MetaParams.EXFILTER_MODEL))
        self.get_param(
            MetaParams.EMFILTER_MODEL_BTN).sigActivated.connect(
                lambda: self.add_param_child(
                    MetaParams.EMFILTER_MODEL_LIST,
                    MetaParams.EMFILTER_MODEL))

        self.get_param(
            MetaParams.IMPORT_STATE).sigActivated.connect(self.load_json)
        self.get_param(
            MetaParams.EXPORT_STATE).sigActivated.connect(self.export_json)
        self.get_param(
            MetaParams.IMPORT_XML).sigActivated.connect(self.load_xml)
        self.get_param(
            MetaParams.EXPORT_XML).sigActivated.connect(self.save)

    def add_param_child(self, parent: MetaParams, value: Union[MetaParams, Any]):
        '''
        Add a child parameter to the specified parent parameter.

        Parameters
        ----------
        parent : MetaParams
            The parent parameter to which the child will be added.
        value : Union[MetaParams, Any]
            The value of the child parameter.

        Returns
        -------
        None
        '''
        parent = self.get_param(parent)
        parent.addChild(
            {'name' : 'Item 1', 'type': 'str',
             'value': self.get_param_value(value) if \
                isinstance(value, MetaParams) else value, 'removable': True},
            True)

    def change(self, param: Parameter, changes: list):
        '''
        Handle parameter changes as needed.

        Parameters
        ----------
        param : Parameter
            The parameter that triggered the change.
        changes : list
            List of changes.

        Returns
        -------
        None
        '''
        # Handle parameter changes as needed
        pass

    def save(self):
        '''
        Save metadata to an OME-XML file.

        Returns
        -------
        None
        '''
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, 'Save metadata', filter='OME-XML Files (*.ome.xml);;')
            if filename:
                ome_obj = self.gen_OME_XML(1, 512, 512)
                with open(filename, 'w', encoding='utf8') as f:
                    f.write(ome_obj.to_xml())
        except Exception as e:
            print(f'Error saving file: {e}')

    def load_xml(self):
        '''
        Load metadata from an OME-XML file.

        Returns
        -------
        None
        '''
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, 'Load metadata', filter='OME-XML Files (*.ome.xml);;')
            if filename:
                xml = ''
                with open(filename, encoding='utf8') as f:
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
        experimenter.first_name = self.get_param_value(MetaParams.EXP_FNAME)
        experimenter.last_name = self.get_param_value(MetaParams.EXP_LNAME)
        experimenter.email = self.get_param_value(MetaParams.EXP_EMAIL)
        experimenter.institution = self.get_param_value(MetaParams.EXP_INSTITUTE)
        ome_obj.experimenters.append(experimenter)

        micro = Microscope()
        micro.manufacturer = self.get_param_value(MetaParams.MICRO_MANUFACTURER)
        micro.model = self.get_param_value(MetaParams.MICRO_MODEL)

        objective = Objective()
        objective.manufacturer = self.get_param_value(MetaParams.OBJ_MANUFACTURER)
        objective.model = self.get_param_value(MetaParams.OBJ_MODEL)
        objective.lens_na = self.get_param_value(MetaParams.OBJ_LENS_NA)
        objective.nominal_magnification = self.get_param_value(MetaParams.OBJ_NOM_MAG)
        objective.immersion = Objective_Immersion._member_map_.get(
            self.get_param_value(MetaParams.OBJ_IMMERSION), Objective_Immersion.OIL)
        objective.correction = Objective_Correction._member_map_.get(
            self.get_param_value(MetaParams.OBJ_CORR), Objective_Correction.PLAN_APO)

        detector = Detector()
        detector.manufacturer = self.get_param_value(MetaParams.DET_MANUFACTURER)
        detector.model = self.get_param_value(MetaParams.DET_MODEL)
        detector.serial_number = self.get_param_value(MetaParams.DET_SERIAL)
        detector.type = Detector_Type._member_map_.get(
            self.get_param_value(MetaParams.DET_TYPE), Detector_Type.CMOS)

        exFilters = Filter()
        exFilters.model = ', '.join(
            self.get_children(MetaParams.EXFILTER_MODEL_LIST))
        emFilters = Filter()
        emFilters.model = ', '.join(
            self.get_children(MetaParams.EMFILTER_MODEL_LIST))

        instrument = Instrument()
        instrument.microscope = micro
        instrument.objectives.append(objective)
        instrument.detectors.append(detector)

        for item in self.get_children(MetaParams.DICHROIC_MODEL_LIST):
            dichroic = Dichroic()
            dichroic.manufacturer = self.get_param_value(
                MetaParams.DICHROIC_MANUFACTURER)
            dichroic.model = item
            instrument.dichroics.append(dichroic)

        instrument.filters.append(exFilters)
        instrument.filters.append(emFilters)

        ome_obj.instruments.append(instrument)

        planes = [om.Plane(
            the_c=0,
            the_t=i,
            the_z=0,
            exposure_time=self.get_param_value(MetaParams.EXPOSURE),
            exposure_time_unit=UnitsTime._member_map_.get(
                self.get_param_value(MetaParams.EXPOSURE_UNIT),
                UnitsTime.SECOND)
        ) for i in range(frames)]

        channel = om.Channel()
        channel.name = self.get_param_value(MetaParams.CHANNEL_NAME)
        channel.fluor = self.get_param_value(MetaParams.FLUOR_NAME)
        channel.acquisition_mode = Channel_AcquisitionMode._member_map_.get(
            self.get_param_value(MetaParams.ACQ_MODE),
            Channel_AcquisitionMode.SINGLE_MOLECULE_IMAGING)
        channel.illumination_type = Channel_IlluminationType._member_map_.get(
            self.get_param_value(MetaParams.ILL_TYPE),
            Channel_IlluminationType.OTHER)
        channel.contrast_method = Channel_ContrastMethod._member_map_.get(
            self.get_param_value(MetaParams.CONTRAST),
            Channel_ContrastMethod.FLUORESCENCE)
        channel.excitation_wavelength = self.get_param_value(MetaParams.EXCITATION)
        channel.emission_wavelength = self.get_param_value(MetaParams.EMISSION)
        channel.excitation_wavelength_unit = UnitsLength._member_map_.get(
            self.get_param_value(MetaParams.WAVE_UNIT), UnitsLength.NANOMETER)
        channel.emission_wavelength_unit = UnitsLength._member_map_.get(
            self.get_param_value(MetaParams.WAVE_UNIT), UnitsLength.NANOMETER)
        channel.samples_per_pixel = 1

        pixels = om.Pixels(
            size_c=channels, size_t=frames,
            size_x=width,
            size_y=height,
            size_z=z_planes,
            type=PixelType._member_map_.get(
                self.get_param_value(MetaParams.PIXEL_TYPE),
                PixelType.UINT16),
            dimension_order=dimension_order,
            # metadata_only=True,
            physical_size_x=self.get_param_value(MetaParams.PX_SIZE),
            physical_size_x_unit=UnitsLength._member_map_.get(
                self.get_param_value(MetaParams.P_UNIT), UnitsLength.MICROMETER),
            physical_size_y=self.get_param_value(MetaParams.PY_SIZE),
            physical_size_y_unit=UnitsLength._member_map_.get(
                self.get_param_value(MetaParams.P_UNIT), UnitsLength.MICROMETER),
            time_increment=self.get_param_value(MetaParams.EXPOSURE),
            time_increment_unit=UnitsTime._member_map_.get(
                self.get_param_value(MetaParams.EXPOSURE_UNIT), UnitsTime.SECOND),
        )
        pixels.tiff_data_blocks.append(om.TiffData())
        pixels.channels.append(channel)
        pixels.planes.extend(planes)
        img = om.Image(
            id='Image:1',
            name=self.get_param_value(MetaParams.EXPERIMENT_NAME),
            pixels=pixels,
            description=self.get_param_value(MetaParams.EXP_DESC)
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
        experimenter.first_name = self.get_param_value(MetaParams.EXP_FNAME)
        experimenter.last_name = self.get_param_value(MetaParams.EXP_LNAME)
        experimenter.email = self.get_param_value(MetaParams.EXP_EMAIL)
        experimenter.institution = self.get_param_value(MetaParams.EXP_INSTITUTE)
        ome_obj.experimenters.append(experimenter)

        detector = Detector()
        detector.manufacturer = self.get_param_value(MetaParams.DET_MANUFACTURER)
        detector.model = self.get_param_value(MetaParams.DET_MODEL)
        detector.serial_number = self.get_param_value(MetaParams.DET_SERIAL)
        detector.type = Detector_Type._member_map_[
            self.get_param_value(MetaParams.DET_TYPE)]

        instrument = Instrument()
        instrument.detectors.append(detector)

        ome_obj.instruments.append(instrument)

        planes = [om.Plane(
            the_c=0,
            the_t=i,
            the_z=0,
            exposure_time=self.get_param_value(MetaParams.EXPOSURE),
            exposure_time_unit=UnitsTime._member_map_[
                self.get_param_value(MetaParams.EXPOSURE_UNIT)
            ]
        ) for i in range(frames)]

        channel = om.Channel()

        pixels = om.Pixels(
            size_c=channels, size_t=frames,
            size_x=width,
            size_y=height,
            size_z=z_planes,
            type=PixelType._member_map_[
                self.get_param_value(MetaParams.PIXEL_TYPE)],
            dimension_order=dimension_order,
            # metadata_only=True,
            physical_size_x=self.get_param_value(MetaParams.PX_SIZE),
            physical_size_x_unit=UnitsLength._member_map_[
                self.get_param_value(MetaParams.P_UNIT)],
            physical_size_y=self.get_param_value(MetaParams.PY_SIZE),
            physical_size_y_unit=UnitsLength._member_map_[
                self.get_param_value(MetaParams.P_UNIT)],
            time_increment=self.get_param_value(MetaParams.EXPOSURE),
            time_increment_unit=UnitsTime._member_map_[
                self.get_param_value(MetaParams.EXPOSURE_UNIT)],
        )
        pixels.tiff_data_blocks.append(om.TiffData())
        pixels.channels.append(channel)
        pixels.planes.extend(planes)
        img = om.Image(
            id='Image:1',
            name=self.get_param_value(MetaParams.EXPERIMENT_NAME),
            pixels=pixels,
            description=self.get_param_value(MetaParams.EXP_DESC)
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
            self.set_param_value(MetaParams.EXPERIMENT_NAME, img.name)
            self.set_param_value(MetaParams.EXP_DESC,img.description)
            if img.pixels is not None:
                pixels = img.pixels
                if pixels.physical_size_x is not None:
                    self.set_param_value(
                        MetaParams.PX_SIZE,
                        float(pixels.physical_size_x))
                    self.set_param_value(
                        MetaParams.P_UNIT,
                        pixels.physical_size_x_unit.name)
                if pixels.physical_size_y is not None:
                    self.set_param_value(
                        MetaParams.PY_SIZE,
                        float(pixels.physical_size_y))
                if pixels.time_increment is not None:
                    self.set_param_value(
                        MetaParams.EXPOSURE,
                        float(pixels.time_increment))
                    self.set_param_value(
                        MetaParams.EXPOSURE_UNIT,
                        pixels.time_increment_unit.name)
                self.set_param_value(
                    MetaParams.PIXEL_TYPE, pixels.type.name)
                if pixels.channels.__len__() > 0:
                    channel = pixels.channels[0]
                    self.set_param_value(MetaParams.CHANNEL_NAME, channel.name)
                    self.set_param_value(MetaParams.FLUOR_NAME, channel.fluor)
                    if channel.acquisition_mode:
                        self.set_param_value(
                            MetaParams.ACQ_MODE,
                            channel.acquisition_mode.name)
                    if channel.illumination_type:
                        self.set_param_value(
                            MetaParams.ILL_TYPE,
                            channel.illumination_type.name)
                    if channel.contrast_method:
                        self.set_param_value(
                            MetaParams.CONTRAST,
                            channel.contrast_method.name)
                    if channel.excitation_wavelength:
                        self.set_param_value(
                            MetaParams.EXCITATION,
                            float(channel.excitation_wavelength))
                        self.set_param_value(
                            MetaParams.WAVE_UNIT,
                            channel.excitation_wavelength_unit.name)
                    if channel.emission_wavelength:
                        self.set_param_value(
                            MetaParams.EMISSION,
                            float(channel.emission_wavelength))
                        self.set_param_value(
                            MetaParams.WAVE_UNIT,
                            channel.emission_wavelength_unit.name)
            if ome_obj.experimenters.__len__() > 0:
                exper = ome_obj.experimenters[0]
                self.set_param_value(MetaParams.EXP_FNAME, exper.first_name)
                self.set_param_value(MetaParams.EXP_LNAME, exper.last_name)
                self.set_param_value(MetaParams.EXP_EMAIL, exper.email)
                self.set_param_value(MetaParams.EXP_INSTITUTE, exper.institution)
            if ome_obj.instruments.__len__() > 0:
                inst = ome_obj.instruments[0]
                if inst.microscope is not None:
                    micro = inst.microscope
                    self.set_param_value(
                        MetaParams.MICRO_MANUFACTURER, micro.manufacturer)
                    self.set_param_value(MetaParams.MICRO_MODEL, micro.model)
                if inst.objectives.__len__() > 0:
                    objective = inst.objectives[0]
                    self.set_param_value(
                        MetaParams.OBJ_MANUFACTURER, objective.manufacturer)
                    self.set_param_value(MetaParams.OBJ_MODEL, objective.model)
                    self.set_param_value(
                        MetaParams.OBJ_LENS_NA,
                        float(objective.lens_na))
                    self.set_param_value(
                        MetaParams.OBJ_NOM_MAG,
                        float(objective.nominal_magnification))
                    self.set_param_value(
                        MetaParams.OBJ_IMMERSION, objective.immersion.name)
                    self.set_param_value(
                        MetaParams.OBJ_CORR, objective.correction.name)
                if inst.detectors.__len__() > 0:
                    detector = inst.detectors[0]
                    self.set_param_value(
                        MetaParams.DET_MANUFACTURER, detector.manufacturer)
                    self.set_param_value(MetaParams.DET_MODEL, detector.model)
                    self.set_param_value(MetaParams.DET_SERIAL, detector.serial_number)
                    self.set_param_value(MetaParams.DET_TYPE, detector.type.name)
                if inst.dichroics.__len__() > 0:
                    self.get_param(MetaParams.DICHROIC_MODEL_LIST).clearChildren()
                    for dichroic in inst.dichroics:
                        self.set_param_value(
                            MetaParams.DICHROIC_MANUFACTURER, dichroic.manufacturer)
                        self.add_param_child(
                            MetaParams.DICHROIC_MODEL_LIST, dichroic.model)
                if inst.filters.__len__() > 1:
                    self.get_param(MetaParams.EXFILTER_MODEL_LIST).clearChildren()
                    self.get_param(MetaParams.EMFILTER_MODEL_LIST).clearChildren()
                    exFilters = inst.filters[0]
                    for _filter in inst.filters[0].model.split(', '):
                        self.add_param_child(MetaParams.EXFILTER_MODEL_LIST, _filter)
                    emFilters = inst.filters[1]
                    for _filter in inst.filters[1].model.split(', '):
                        self.add_param_child(MetaParams.EMFILTER_MODEL_LIST, _filter)
