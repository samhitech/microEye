from typing import Callable, Union

import numpy as np

from microEye.qt import QApplication, QtCore, QtWidgets, Signal


def create_line_edit(
    label,
    default_text,
    layout: Union[QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout],
) -> QtWidgets.QLineEdit:
    '''
    Create a QLineEdit widget with label.

    Parameters
    ----------
    label : str
        The label for the line edit.
    default_text : str
        The default text to display in the line edit.
    layout : Union[QVBoxLayout, QHBoxLayout, QFormLayout]
        The layout to which the line edit will be added.

    Returns
    -------
    QLineEdit
        The created QLineEdit widget.
    '''
    line_edit = QtWidgets.QLineEdit(default_text)

    if isinstance(layout, QtWidgets.QFormLayout):
        layout.addRow(QtWidgets.QLabel(label), line_edit)
    else:
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(line_edit)

    return line_edit


def create_text_edit(
    label,
    default_text,
    layout: Union[QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout],
) -> QtWidgets.QTextEdit:
    '''
    Create a QLineEdit widget with label.

    Parameters
    ----------
    label : str
        The label for the line edit.
    default_text : str
        The default text to display in the line edit.
    layout : Union[QVBoxLayout, QHBoxLayout, QFormLayout]
        The layout to which the line edit will be added.

    Returns
    -------
    QLineEdit
        The created QLineEdit widget.
    '''
    text_edit = QtWidgets.QTextEdit(default_text)
    if isinstance(layout, QtWidgets.QFormLayout):
        layout.addRow(QtWidgets.QLabel(label))
        layout.addRow(text_edit)
    else:
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(text_edit)
    return text_edit


def create_combo_box(
    label,
    items,
    default_item,
    layout: Union[QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout],
) -> QtWidgets.QComboBox:
    '''
    Create a QComboBox widget with label.

    Parameters
    ----------
    label : str
        The label for the combo box.
    items : list
        The list of items to populate the combo box.
    default_item : str
        The default item to select in the combo box.
    layout : Union[QVBoxLayout, QHBoxLayout, QFormLayout]
        The layout to which the combo box will be added.

    Returns
    -------
    QComboBox
        The created QComboBox widget.
    '''
    combo_box = QtWidgets.QComboBox()
    combo_box.addItems(items)
    combo_box.setCurrentText(default_item)
    if isinstance(layout, QtWidgets.QFormLayout):
        layout.addRow(QtWidgets.QLabel(label), combo_box)
    else:
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(combo_box)
    return combo_box


def create_double_spin_box(
    min_value=0, max_value=1, single_step=0.01, decimals=3, initial_value=0, slot=None
):
    '''
    Create a QDoubleSpinBox widget.

    Parameters
    ----------
    min_value : float, optional
        Minimum allowed value.
    max_value : float, optional
        Maximum allowed value.
    single_step : float, optional
        Step size for increment/decrement.
    decimals : int, optional
        Number of decimal places.
    initial_value : float, optional
        Initial value of the spin box.
    slot : Callable, optional
        Function to connect to the valueChanged signal.

    Returns
    -------
    QDoubleSpinBox
        The created QDoubleSpinBox widget.
    '''
    double_spin_box = QtWidgets.QDoubleSpinBox()
    double_spin_box.setMinimum(min_value)
    double_spin_box.setMaximum(max_value)
    double_spin_box.setSingleStep(single_step)
    double_spin_box.setDecimals(decimals)
    double_spin_box.setValue(initial_value)
    if slot:
        double_spin_box.valueChanged.connect(slot)
    return double_spin_box


def create_labelled_double_spin_box(
    label,
    min_val,
    max_val,
    default_val,
    layout: Union[QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout],
) -> QtWidgets.QDoubleSpinBox:
    '''
    Create a QDoubleSpinBox widget with a label.

    Parameters
    ----------
    label : str
        The label for the spin box.
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.
    default_val : float
        Initial value of the spin box.
    layout : Union[QVBoxLayout, QHBoxLayout, QFormLayout]
        The layout to which the spin box will be added.

    Returns
    -------
    QDoubleSpinBox
        The created QDoubleSpinBox widget.
    '''
    spin_box = QtWidgets.QDoubleSpinBox()
    spin_box.setMinimum(min_val)
    spin_box.setMaximum(max_val)
    spin_box.setValue(default_val)

    if isinstance(layout, QtWidgets.QFormLayout):
        layout.addRow(QtWidgets.QLabel(label), spin_box)
    else:
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(spin_box)
    return spin_box


def create_spin_box(
    min_value=0, max_value=100, single_step=1, initial_value=0, slot=None
):
    '''
    Create a QSpinBox widget.

    Parameters
    ----------
    min_value : int, optional
        Minimum allowed value.
    max_value : int, optional
        Maximum allowed value.
    single_step : int, optional
        Step size for increment/decrement.
    initial_value : int, optional
        Initial value of the spin box.
    slot : Callable, optional
        Function to connect to the valueChanged signal.

    Returns
    -------
    QSpinBox
        The created QSpinBox widget.
    '''

    spin_box = QtWidgets.QSpinBox()
    spin_box.setMinimum(min_value)
    spin_box.setMaximum(max_value)
    spin_box.setSingleStep(single_step)
    spin_box.setValue(initial_value)
    if slot:
        spin_box.valueChanged.connect(slot)
    return spin_box


def create_check_box(text, initial_state=False, state_changed_slot=None):
    '''
    Create a QCheckBox widget.

    Parameters
    ----------
    text : str
        Text to display next to the checkbox.
    initial_state : bool, optional
        Initial state of the checkbox.
    state_changed_slot : Callable, optional
        Function to connect to the stateChanged signal.

    Returns
    -------
    QCheckBox
        The created QCheckBox widget.
    '''
    check_box = QtWidgets.QCheckBox(text)
    check_box.setChecked(initial_state)

    if state_changed_slot:
        check_box.stateChanged.connect(state_changed_slot)

    return check_box


def create_group_box(
    title,
    layout_type: Union[
        QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout
    ] = QtWidgets.QVBoxLayout,
):
    '''
    Create a QGroupBox with a specified layout type.

    Parameters
    ----------
    title : str
        Title of the group box.
    layout_type : QVBoxLayout or QHBoxLayout or QFormLayout, optional
        Type of layout to use inside the group box.

    Returns
    -------
    QGroupBox, QLayout
        The created QGroupBox and its associated layout.
    '''
    group_box = QtWidgets.QGroupBox(title)
    layout = layout_type()
    group_box.setLayout(layout)
    return group_box, layout


def create_widget(
    layout_type: Union[
        QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout
    ] = QtWidgets.QVBoxLayout,
) -> tuple[
    QtWidgets.QWidget,
    Union[QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout],
]:
    '''
    Create a QWidget with a specified layout type.

    Parameters
    ----------
    layout_type : QVBoxLayout or QHBoxLayout or QFormLayout, optional
        Type of layout to use inside the widget.

    Returns
    -------
    QWidget, QLayout
        The created QWidget and its associated layout.
    '''
    widget = QtWidgets.QWidget()
    layout = layout_type()
    widget.setLayout(layout)
    return widget, layout


def create_hbox_layout(*args):
    '''
    Create a QHBoxLayout and add the specified widgets to it.

    Parameters
    ----------
    *args : QWidget
        Widgets to add to the layout.

    Returns
    -------
    QHBoxLayout
        The created QHBoxLayout.
    '''
    hbox_layout = QtWidgets.QHBoxLayout()

    for widget in args:
        hbox_layout.addWidget(widget)

    return hbox_layout


def create_vbox_layout(*args):
    '''
    Create a QVBoxLayout and add the specified widgets to it.

    Parameters
    ----------
    *args : QWidget
        Widgets to add to the layout.

    Returns
    -------
    QVBoxLayout
        The created QVBoxLayout.
    '''
    vbox_layout = QtWidgets.QVBoxLayout()

    for widget in args:
        vbox_layout.addWidget(widget)

    return vbox_layout


def layout_add_elements(
    layout: Union[QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout], *args
):
    '''
    Add elements to a QVBoxLayout or QHBoxLayout.

    Parameters
    ----------
    layout : QVBoxLayout or QHBoxLayout
        The layout to which elements will be added.
    *args : QWidget or QLayout
        Elements to add to the layout.
    '''
    for widget in args:
        if isinstance(widget, QtWidgets.QWidget):
            layout.addWidget(widget)
        elif isinstance(widget, QtWidgets.QLayout):
            layout.addLayout(widget)


def get_scaling_factor(height, width, target_percentage=0.8):
    '''
    Calculate the scaling factor to fill the specified percentage of the screen.

    Parameters
    ----------
    height : int
        Height of the content.
    width : int
        Width of the content.
    target_percentage : float, optional
        Target percentage of the screen to fill.

    Returns
    -------
    float
        The calculated scaling factor.
    '''
    main_screen = QApplication.primaryScreen().availableGeometry()
    screen_info = (main_screen.width(), main_screen.height())

    # Calculate scaling factor to fill the specified percentage of the screen
    target_height = screen_info[1] * target_percentage
    target_width = screen_info[0] * target_percentage

    scaling_factor_height = target_height / height
    scaling_factor_width = target_width / width

    return min(scaling_factor_height, scaling_factor_width)


def time_string_ms(value: float):
    '''
    Convert time in milliseconds to a formatted string.

    Parameters
    ----------
    value : float
        Time value in milliseconds.

    Returns
    -------
    str
        Formatted time string in minutes and seconds.
    '''
    seconds = value / 1000
    return f'{seconds // 60:.0f} min {seconds % 60} seconds'


def time_string_s(value: float):
    '''
    Convert time in seconds to a formatted string.

    Parameters
    ----------
    value : float
        Time value in seconds.

    Returns
    -------
    str
        Formatted time string in minutes and seconds.
    '''
    return f'{value // 60:.0f} min {value % 60} seconds'


def GaussianOffSet(x, a, x0, sigma, offset):
    '''
    Returns a Gaussian function with an offset.

    f(x) = a * exp(-(x - x0)**2 / (2 * sigma**2)) + offset

    Parameters
    ----------
    x : np.ndarray
        Input values.
    a : float
        Amplitude.
    x0 : float
        Center or mean.
    sigma : float
        Standard deviation.
    offset : float
        Y offset.

    Returns
    -------
    np.ndarray
        Gaussian function f(x).
    '''
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


def debounceSlot(parent: QtCore.QObject, signal: Signal, callback: Callable, wait=200):
    '''
    Debounce a slot to limit the frequency of signal emissions.

    Parameters
    ----------
    signal : Signal
        The signal to debounce.
    callback : Callable
        The function to call when the signal is emitted.
    wait : int, optional
        Wait time in milliseconds before calling the callback.

    Returns
    -------
    QTimer
        The timer used for debouncing.
    '''
    timer = QtCore.QTimer(parent)
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: callback())

    signal.connect(lambda: timer.start(wait))

    return timer
