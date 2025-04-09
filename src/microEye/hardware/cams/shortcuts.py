from typing import Union

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.micam import miCamera
from microEye.qt import QDateTime, Qt, QtCore, QtGui, QtWidgets, Signal, Slot
from microEye.utils.gui_helper import debounceSlot

DEFAULT_EXPOSURE_SHORTCUTS = [
    1,
    3,
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    100,
    150,
    200,
    300,
    500,
]


class DiscreteSlider(QtWidgets.QSlider):
    '''
    A QSlider that snaps to discrete, predefined values.
    '''

    def __init__(self, values: list, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._values = values

        self.__font_size = 9

        # The slider's range is 0..(len(values)-1).
        self.setMinimum(0)
        self.setMaximum(len(values) - 1)

        # Dynamically calculate the minimum width
        # based on the largest value and number of values
        font = self.font()
        font.setPointSize(self.__font_size)
        fm = QtGui.QFontMetrics(font)

        max_label_width = max(fm.horizontalAdvance(str(val)) for val in self._values)
        spacing = 5  # Add some spacing between ticks
        self.setMinimumWidth(len(self._values) * (max_label_width + spacing))

        self.setMinimumHeight(60)

        # Make ticks visible (optional)
        self.setTickPosition(QtWidgets.QSlider.TickPosition.TicksAbove)
        self.setTickInterval(1)

        # Optionally, you could make it snap to ticks:
        self.setSingleStep(1)
        self.setPageStep(1)

    def valueFromIndex(self):
        '''
        Returns the label corresponding to the current slider value.
        '''
        return self._values[self.value()]

    def setIndexFromValue(self, val):
        '''
        Given a label from the predefined list, move the slider to that position.
        '''
        if val in self._values:
            idx = self._values.index(val)
            self.setValue(idx)

    def paintEvent(self, event):
        # First, draw the slider normally.
        super().paintEvent(event)

        # Create a QPainter for custom drawing.
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # change painter font size to smaller than the default
        font = painter.font()
        font.setPointSize(self.__font_size)
        painter.setFont(font)

        # Prepare a style option so we can get accurate geometry.
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)

        # We’ll use the handle rect to get the precise x-coord for each step
        fm = QtGui.QFontMetrics(painter.font())

        # For each discrete value
        for i, label in enumerate(self._values):
            # Tell the style we want geometry for sliderPosition = i
            option.sliderPosition = i

            if i == 0:
                label = 'min'

            # Get the handle rect for that position
            handle_rect = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderHandle,
                self,
            )

            # Center x on the handle
            x_center = handle_rect.center().x()

            # Decide where you want to place the text vertically:
            #   - Above the handle?   -> handle_rect.top() - some_offset
            #   - Below the groove?   -> handle_rect.bottom() + some_offset
            # Adjust as needed:
            y_text = handle_rect.top() - 4

            text_width = fm.horizontalAdvance(str(label))
            text_height = fm.height()

            # Build a small bounding rect so we can center-align text
            rect = QtCore.QRectF(
                x_center - text_width / 2,  # left
                y_text - text_height,  # top
                text_width,
                text_height,
            )

            # Optionally clamp rect within the widget’s boundaries
            if rect.left() < 0:
                rect.moveLeft(0)
            if rect.right() > self.width():
                rect.moveRight(self.width())

            # Draw the label, centered in our bounding rect
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(label))

        painter.end()


class CameraShortcutsWidget(QtWidgets.QWidget):
    '''
    A widget containing camera shortcuts for common operations like exposure adjustment
    and toggling autostretch.
    '''

    exposureChanged = Signal(float)
    autostretchChanged = Signal()
    previewChanged = Signal()
    displayStatsChanged = Signal()
    displayModeChanged = Signal(str)
    saveDataChanged = Signal()
    zoomChanged = Signal(float)
    acquisitionStart = Signal()
    acquisitionStop = Signal()
    adjustName = Signal(str)

    def __init__(
        self,
        camera: miCamera = None,
        exposure_range=None,
        exposure_shortcuts=DEFAULT_EXPOSURE_SHORTCUTS,
        parent=None,
    ):
        '''
        Initialize the camera shortcuts widget.

        Parameters
        ----------
        camera_options : CameraOptions
            The camera options object for callbacks
        camera : miCamera, optional
            Camera object for accessing properties like exposure range
        exposure_range : tuple, optional
            Min and max exposure values as (min, max)
        exposure_shortcuts : list, optional
            List of exposure shortcut values
        parent : QWidget, optional
            Parent widget
        '''
        super().__init__(parent)

        self._exposure_shortcuts = exposure_shortcuts or DEFAULT_EXPOSURE_SHORTCUTS

        # If camera is provided, use its exposure range
        if camera is not None and hasattr(camera, 'exposure_range'):
            self.min_exposure = camera.exposure_range[0]
        elif exposure_range is not None:
            self.min_exposure = exposure_range[0]
        else:
            self.min_exposure = 0.1  # Default minimum exposure

        # Set up the UI
        self._setup_ui()

    def _setup_ui(self):
        """Set up the widget's UI components"""
        self.__layout = QtWidgets.QFormLayout(self)
        self.setLayout(self.__layout)

        # Exposure Time shortcuts
        self.exposure_slider = DiscreteSlider(
            [self.min_exposure, *self._exposure_shortcuts]
        )

        self._debounce_timer = debounceSlot(
            self,
            self.exposure_slider.valueChanged,
            lambda: self.exposureChanged.emit(self.exposure_slider.valueFromIndex()),
            200,
        )

        self.exposure_slider.setIndexFromValue(50.0)

        self.__layout.addRow('Exposure [ms]: ', self.exposure_slider)

        # Zoom slider
        self.display_size_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.display_size_slider.setMinimum(10)
        self.display_size_slider.setMaximum(400)
        self.display_size_slider.setValue(50)
        self.display_size_slider.valueChanged.connect(
            lambda: self.zoomChanged.emit(self.display_size_slider.value() / 100)
        )

        self.__layout.addRow('Zoom [%]: ', self.display_size_slider)

        # Add a layout for experiment name buttons
        experiment_name_btns = QtWidgets.QHBoxLayout()
        self.__layout.addRow('Experiment Name:', experiment_name_btns)

        # Increament index
        self.increment_index_button = QtWidgets.QPushButton('+ Index')
        self.increment_index_button.setToolTip('Increment the current index')
        self.increment_index_button.clicked.connect(
            lambda: self.adjustName.emit('index')
        )

        experiment_name_btns.addWidget(self.increment_index_button)

        # unique index
        self.unique_index_button = QtWidgets.QPushButton('Unique Index')
        self.unique_index_button.setToolTip('Set a unique index')
        self.unique_index_button.clicked.connect(
            lambda: self.adjustName.emit('unique_index')
        )

        experiment_name_btns.addWidget(self.unique_index_button)

        # Add exposure to name
        self.add_exposure_button = QtWidgets.QPushButton('+ Exposure')
        self.add_exposure_button.setToolTip('Add the current exposure to the name')
        self.add_exposure_button.clicked.connect(
            lambda: self.adjustName.emit('exposure')
        )

        experiment_name_btns.addWidget(self.add_exposure_button)

        # buttons
        acq_layout = QtWidgets.QHBoxLayout()
        self.__layout.addRow('Acquisition Options:', acq_layout)

        # Start/Stop acquisition
        self.acquisition_button = QtWidgets.QPushButton('Start Acquisition')
        self.acquisition_button.clicked.connect(self.acquisitionStart.emit)

        acq_layout.addWidget(self.acquisition_button)

        self.stop_button = QtWidgets.QPushButton('Stop Acquisition')
        self.stop_button.clicked.connect(self.acquisitionStop.emit)

        acq_layout.addWidget(self.stop_button)

        # Toggle save data
        self.save_data_button = QtWidgets.QCheckBox('Save Data')
        self.save_data_button.clicked.connect(self.saveDataChanged.emit)

        acq_layout.addWidget(self.save_data_button)

        # buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.__layout.addRow('Display Options:', button_layout)

        # Toggle autostretch
        self.autostretch_button = QtWidgets.QCheckBox('Autostretching')
        self.autostretch_button.clicked.connect(self.autostretchChanged.emit)

        button_layout.addWidget(self.autostretch_button)

        # Toggle preview
        self.preview_button = QtWidgets.QCheckBox('Preview')
        self.preview_button.clicked.connect(self.previewChanged.emit)

        button_layout.addWidget(self.preview_button)

        # Stats preview
        self.stats_button = QtWidgets.QCheckBox('Display Stats')
        self.stats_button.clicked.connect(self.displayStatsChanged.emit)

        button_layout.addWidget(self.stats_button)

        # Display mode 4 radios
        modes_layout = QtWidgets.QHBoxLayout()
        self.__layout.addRow('Display Mode:', modes_layout)

        self.display_mode = QtWidgets.QButtonGroup(self)
        self.display_mode.setExclusive(True)
        self.display_mode.setObjectName('Display Mode')

        self.display_modes = [
            str(CamParams.SINGLE_VIEW),
            str(CamParams.DUAL_SIDE),
            str(CamParams.DUAL_OVERLAID),
            str(CamParams.ROIS_VIEW),
        ]
        for idx, mode in enumerate(self.display_modes):
            radio_button = QtWidgets.QRadioButton(mode)
            radio_button.setObjectName(str(mode))
            self.display_mode.addButton(radio_button, idx)
            modes_layout.addWidget(radio_button)

        self.display_mode.idClicked.connect(self._display_mode_changed)

    def _display_mode_changed(self, index):
        '''
        Handle display mode changes

        Parameters
        ----------
        index : int
            The ID of the selected radio button
        '''
        mode = self.display_modes[index]
        self.displayModeChanged.emit(mode)

    def set_exposure(self, value):
        '''Set the exposure slider to the given value'''
        self.exposure_slider.setIndexFromValue(value)

    def get_exposure(self):
        '''Get the current exposure value from the slider'''
        return self.exposure_slider.valueFromIndex()

    def set_zoom(self, value):
        '''Set the zoom slider to the given value'''
        self.display_size_slider.setValue(int(value * 100))

    def get_zoom(self):
        '''Get the current zoom value from the slider'''
        return self.display_size_slider.value() / 100

    def set_autostretch(self, value):
        '''Set the autostretch checkbox to the given value'''
        self.autostretch_button.setChecked(value)

    def set_preview(self, value):
        '''Set the preview checkbox to the given value'''
        self.preview_button.setChecked(value)

    def set_display_stats(self, value):
        '''Set the display stats checkbox to the given value'''
        self.stats_button.setChecked(value)

    def set_display_mode(self, mode: Union[CamParams, str]):
        '''Set the display mode radio button to the given value'''
        if isinstance(mode, CamParams):
            mode = str(mode)

        if mode in self.display_modes:
            button = self.display_mode.button(self.display_modes.index(mode))
            if button:
                button.setChecked(True)

    def set_save_data(self, value):
        '''Set the save data checkbox to the given value'''
        self.save_data_button.setChecked(value)

    def create_context_menu(self, parent):
        '''
        Create a context menu containing this widget

        Parameters
        ----------
        parent : QWidget
            The parent widget for the menu

        Returns
        -------
        QMenu
            The created context menu
        '''
        menu = QtWidgets.QMenu(parent)
        widget_action = QtWidgets.QWidgetAction(menu)
        widget_action.setDefaultWidget(self)
        menu.addAction(widget_action)
        return menu
