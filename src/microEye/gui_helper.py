from typing import Union

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSpinBox,
    QVBoxLayout,
)


def create_double_spin_box(
        min_value=0, max_value=1, single_step=0.01,
        decimals=3, initial_value=0, slot=None):
    double_spin_box = QDoubleSpinBox()
    double_spin_box.setMinimum(min_value)
    double_spin_box.setMaximum(max_value)
    double_spin_box.setSingleStep(single_step)
    double_spin_box.setDecimals(decimals)
    double_spin_box.setValue(initial_value)
    if slot:
        double_spin_box.valueChanged.connect(slot)
    return double_spin_box

def create_spin_box(
        min_value=0, max_value=100,
        single_step=1, initial_value=0, slot=None):
    spin_box = QSpinBox()
    spin_box.setMinimum(min_value)
    spin_box.setMaximum(max_value)
    spin_box.setSingleStep(single_step)
    spin_box.setValue(initial_value)
    if slot:
        spin_box.valueChanged.connect(slot)
    return spin_box

def create_check_box(text, initial_state=False, state_changed_slot=None):
    check_box = QCheckBox(text)
    check_box.setChecked(initial_state)

    if state_changed_slot:
        check_box.stateChanged.connect(state_changed_slot)

    return check_box

def create_group_box(
        title, layout_type: Union[QVBoxLayout, QHBoxLayout, QFormLayout] = QVBoxLayout):
    group_box = QGroupBox(title)
    layout = layout_type()
    group_box.setLayout(layout)
    return group_box, layout

def create_hbox_layout(*args):
    hbox_layout = QHBoxLayout()

    for widget in args:
        hbox_layout.addWidget(widget)

    return hbox_layout

def get_scaling_factor(height, width, target_percentage=0.8):
    app: QApplication = QApplication.instance()
    desktop = app.desktop()

    main_screen = desktop.screenGeometry()
    screen_info = (main_screen.width(), main_screen.height())

    # Calculate scaling factor to fill the specified percentage of the screen
    target_height = screen_info[1] * target_percentage
    target_width = screen_info[0] * target_percentage

    scaling_factor_height = target_height / height
    scaling_factor_width = target_width / width

    return min(scaling_factor_height, scaling_factor_width)
