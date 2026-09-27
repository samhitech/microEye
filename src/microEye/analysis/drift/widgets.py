from microEye.qt import Qt, QtCore, QtGui, QtWidgets, Signal


class DriftOptionsDialog(QtWidgets.QDialog):
    '''
    Dialog for configuring and executing drift correction options dynamically.
    Emits method_name and kwargs via signal on accept.
    '''

    # Signal emitting (method_name: str, kwargs: dict)
    drift_requested = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Drift Correction Options')
        self.setModal(True)
        self.resize(420, 320)

        # Main Layout
        self._layout = QtWidgets.QVBoxLayout(self)

        # 1. Method Selection Area
        method_layout = QtWidgets.QHBoxLayout()
        self.method_label = QtWidgets.QLabel('Drift Correction Method:')
        self.method_combo = QtWidgets.QComboBox()

        # Mapping Combo Text -> Backend Method Identifier
        self.methods = {
            'COMET': 'comet',
            'AIM Drift Correction': 'aim',
            'RCC (Redundant Cross-Corr)': 'rcc',
            'DCC (Direct Cross-Corr)': 'dcc',
            'MCC (Mean Cross-Corr)': 'mcc',
            'Phase Cross-Corr': 'phase',
            'Fiducials Drift Correction (FDM)': 'fdm',
        }
        self.method_combo.addItems(list(self.methods.keys()))
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)
        self._layout.addLayout(method_layout)

        # 2. Dynamic Stacked Widget for Page Parameters
        self.stacked_widget = QtWidgets.QStackedWidget()
        self._layout.addWidget(self.stacked_widget)

        # Create parameter UI panels
        self.__init_cross_corr_ui()
        self.__init_aim_ui()
        self.__init_comet_ui()

        # Empty placeholder page for unrecognized methods
        self.empty_widget = QtWidgets.QWidget()
        self.stacked_widget.addWidget(self.empty_widget)

        # Connect Combo Switch Event
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

        # 3. Dialog Action Buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        self._layout.addWidget(self.button_box)

        # Initialize to first page view
        self._on_method_changed()

    # --- UI Panel Builders ---

    def __init_cross_corr_ui(self):
        '''Shared form page for Cross-Correlation methods (RCC, DCC, MCC, Phase, FDM)'''
        self.cross_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.cross_widget)

        self.cross_bins = QtWidgets.QSpinBox()
        self.cross_bins.setValue(10)

        self.cross_px = QtWidgets.QSpinBox()
        self.cross_px.setValue(10)

        self.cross_up = QtWidgets.QSpinBox()
        self.cross_up.setRange(0, 1000)
        self.cross_up.setValue(100)

        self.cross_rmax = QtWidgets.QSpinBox()
        self.cross_rmax.setRange(0, 1000)
        self.cross_rmax.setValue(23)

        form.addRow('Time Bins:', self.cross_bins)
        form.addRow('Pixel Size (nm):', self.cross_px)
        form.addRow('Upsampling Factor:', self.cross_up)
        form.addRow('R Max:', self.cross_rmax)

        self.stacked_widget.addWidget(self.cross_widget)

    def __init_aim_ui(self):
        '''Form page for AIM Drift Correction'''
        self.aim_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.aim_widget)

        self.aim_segment_size = QtWidgets.QSpinBox()
        self.aim_segment_size.setRange(1, 10000)
        self.aim_segment_size.setValue(100)

        self.aim_intersect_d = QtWidgets.QDoubleSpinBox()
        self.aim_intersect_d.setRange(1.0, 1000.0)
        self.aim_intersect_d.setValue(20.0)

        self.aim_z_scale = QtWidgets.QDoubleSpinBox()
        self.aim_z_scale.setRange(0.01, 10.0)
        self.aim_z_scale.setValue(1.0)

        form.addRow('Segment Size:', self.aim_segment_size)
        form.addRow('Intersect Distance:', self.aim_intersect_d)
        form.addRow('Z Scale:', self.aim_z_scale)

        self.stacked_widget.addWidget(self.aim_widget)

    def __init_comet_ui(self):
        '''Form page for COMET Drift Correction

        max_drift : float, optional
            Maximum expected drift in physical units (e.g. nm). Default is 300.0.
        locs_per_window : int, optional
            Target number of localizations per time segment.
        downsample_factor : int, optional
            Factor by which to downsample the dataset for coarse drift estimation.
        interp_kind : str, optional
            Interpolation method for per-frame drift estimation.
        sigma_start : float, optional
            Starting Gaussian length scale for COMET optimization.
            Default is 40.0.
        sigma_end : float, optional
            Ending Gaussian length scale for COMET optimization.
            Default is 5.0.
        sigma_steps : int, optional
            Number of Gaussian length scales to use in COMET optimization.
            Default is 4.
        boxcar_width : int, optional
            Width of boxcar smoothing applied to the coarse drift estimate.
            Default is 3. To disable smoothing, set to 1.
        '''
        self.comet_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.comet_widget)

        self.comet_max_drift = QtWidgets.QDoubleSpinBox()
        self.comet_max_drift.setRange(0.0, 10000.0)
        self.comet_max_drift.setValue(300.0)

        self.comet_locs_per_window = QtWidgets.QSpinBox()
        self.comet_locs_per_window.setRange(1, 100000)
        self.comet_locs_per_window.setValue(500)

        self.comet_sigma_start = QtWidgets.QDoubleSpinBox()
        self.comet_sigma_start.setRange(1.0, 1000.0)
        self.comet_sigma_start.setValue(40.0)

        self.comet_sigma_end = QtWidgets.QDoubleSpinBox()
        self.comet_sigma_end.setRange(1.0, 1000.0)
        self.comet_sigma_end.setValue(5.0)

        self.comet_sigma_steps = QtWidgets.QSpinBox()
        self.comet_sigma_steps.setRange(1, 100)
        self.comet_sigma_steps.setValue(4)

        self.comet_downsample_factor = QtWidgets.QDoubleSpinBox()
        self.comet_downsample_factor.setRange(0.01, 10000.0)
        self.comet_downsample_factor.setValue(1.0)
        self.comet_downsample_factor.setDecimals(2)

        self.comet_boxcar_width = QtWidgets.QSpinBox()
        self.comet_boxcar_width.setRange(1, 100)
        self.comet_boxcar_width.setValue(3)
        # make sure always odd boxcar width
        def on_boxcar_changed(val):
            if val % 2 == 0:
                self.comet_boxcar_width.setValue(val + 1)
        self.comet_boxcar_width.valueChanged.connect(lambda val: on_boxcar_changed(val))



        form.addRow('Max Drift [nm]:', self.comet_max_drift)
        form.addRow('Points per Window:', self.comet_locs_per_window)
        form.addRow('Sigma Start [nm]:', self.comet_sigma_start)
        form.addRow('Sigma End [nm]:', self.comet_sigma_end)
        form.addRow('Sigma Steps:', self.comet_sigma_steps)
        form.addRow('Downsample Factor:', self.comet_downsample_factor)
        form.addRow('Boxcar Width:', self.comet_boxcar_width)

        self.stacked_widget.addWidget(self.comet_widget)

    # --- Switcher Logic ---

    def _on_method_changed(self):
        '''Swaps the current stacked parameter page depending on combo option.'''
        selected_key = self.method_combo.currentText()
        method_id = self.methods[selected_key]

        if method_id in ['rcc', 'dcc', 'mcc', 'phase']:
            self.stacked_widget.setCurrentWidget(self.cross_widget)
        elif method_id == 'aim':
            self.stacked_widget.setCurrentWidget(self.aim_widget)
        elif method_id == 'comet':
            self.stacked_widget.setCurrentWidget(self.comet_widget)
        else:
            # empty page for unrecognized methods
            self.stacked_widget.setCurrentWidget(self.empty_widget)

    # --- Data Packaging & Signals ---

    def get_kwargs(self, method_id: str = None) -> tuple[str, dict]:
        '''Extracts the method string and parameter dictionary for execution.'''
        if method_id is None:
            selected_key = self.method_combo.currentText()
            method_id = self.methods[selected_key]

        kwargs = {}

        if method_id in ['rcc', 'dcc', 'mcc', 'phase']:
            kwargs = {
                'n_bins': self.cross_bins.value(),
                'pixel_size': self.cross_px.value(),
                'upsampling': self.cross_up.value(),
                'rmax': self.cross_rmax.value(),
            }
        elif method_id == 'aim':
            kwargs = {
                'segment_size': self.aim_segment_size.value(),
                'intersect_d': self.aim_intersect_d.value(),
                'z_scale': self.aim_z_scale.value(),
            }
        elif method_id == 'comet':
            kwargs = {
                'max_drift': self.comet_max_drift.value(),
                'locs_per_window': self.comet_locs_per_window.value(),
                'sigma_start': self.comet_sigma_start.value(),
                'sigma_end': self.comet_sigma_end.value(),
                'sigma_steps': self.comet_sigma_steps.value(),
                'downsample_factor': self.comet_downsample_factor.value(),
                'boxcar_width': self.comet_boxcar_width.value(),
            }

        return method_id, kwargs

    def _on_accept(self):
        '''Handles OK button click, emits signal, and closes dialog.'''
        method_id, kwargs = self.get_kwargs()
        self.drift_requested.emit(method_id, kwargs)
        self.accept()


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    dialog = DriftOptionsDialog()
    dialog.show()
    sys.exit(app.exec())
