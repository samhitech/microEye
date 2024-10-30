import json
import os
from dataclasses import asdict
from pprint import pprint
from typing import Union

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn
from tabulate import tabulate

from microEye.analysis.fitting.psf import ConfidenceMethod, PSFdata, stats
from microEye.analysis.fitting.results import PARAMETER_HEADERS, FittingMethod
from microEye.qt import QtCore, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.enum_encoder import EnumEncoder


class PSFView(QtWidgets.QWidget):
    '''
    A class for viewing and interacting with PSF data
    '''

    def __init__(self, psf_data: Union['PSFdata', str] = None):
        '''Initialize the PSFView.

        Parameters
        ----------
        psf_data : PSFdata or str, optional
            PSF data to visualize. Can be a PSFdata object or a path to an HDF5 file
        '''
        super().__init__()

        if isinstance(psf_data, str) and os.path.exists(psf_data):
            self.psf_data = PSFdata.load_hdf(psf_data)
        elif isinstance(psf_data, PSFdata):
            self.psf_data = psf_data
        else:
            raise TypeError(
                'PSF data must be a PSFdata object or a path to an HDF5 file.'
            )

        self.fit_result = None

        self.setWindowTitle('PSF Data Viewer')
        self._threadpool = QtCore.QThreadPool.globalInstance()
        self.main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.main_layout)

        # Tab Widget for different views
        self.view_tabs = QtWidgets.QTabWidget()
        self.main_layout.addWidget(self.view_tabs)

        # Default view mode
        self.view_mode = 'XY'

        # Setup different view tabs
        self.setup_merged_view_tab()
        self.setup_3d_view_tab()
        self.setup_statistics_tab()

        if psf_data:
            self.update_all_views(False)

    def setup_merged_view_tab(self):
        '''Set up the merged Slice/Field View tab with enhanced controls.'''
        merged_widget = QtWidgets.QWidget()
        merged_layout = QtWidgets.QHBoxLayout()
        merged_widget.setLayout(merged_layout)

        # Visualization area
        visual_layout = QtWidgets.QVBoxLayout()

        self.image_widget = pg.GraphicsLayoutWidget()
        self.view_box: pg.ViewBox = self.image_widget.addViewBox(row=0, col=0)
        self.view_box.setAspectLocked(True)
        self.view_box.setAutoVisible(True)
        self.view_box.enableAutoRange()
        self.view_box.invertY(True)

        # Create two image items: one for PSF data and one for grid overlay
        self.psf_image_item: pg.ImageItem = pg.ImageItem(axisOrder='row-major')
        self.grid_overlay_item: pg.ImageItem = pg.ImageItem(axisOrder='row-major')
        self.view_box.addItem(self.psf_image_item)
        self.view_box.addItem(self.grid_overlay_item)

        # Create histogram with enhanced controls
        self.histogram = pg.HistogramLUTItem(
            gradientPosition='right', orientation='vertical'
        )
        self.histogram.setImageItem(self.psf_image_item)
        self.histogram.gradient.setColorMap(pg.colormap.get('jet', 'matplotlib'))
        self.image_widget.addItem(self.histogram, row=0, col=1)

        visual_layout.addWidget(self.image_widget)

        # Modify Z-slice selection to be more generic
        slice_layout = QtWidgets.QHBoxLayout()
        self.slice_label = QtWidgets.QLabel('Z-Slice:')
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice_spinbox = QtWidgets.QSpinBox()
        self.slice_slider.setMinimum(0)
        self.slice_spinbox.setMinimum(0)
        self.slice_spinbox.setMinimumWidth(75)

        max_slice = len(self.psf_data) - 1
        self.slice_slider.setMaximum(max_slice)
        self.slice_spinbox.setMaximum(max_slice)

        self.slice_slider.valueChanged.connect(self.slice_spinbox.setValue)
        self.slice_spinbox.valueChanged.connect(self.slice_slider.setValue)
        self.slice_slider.valueChanged.connect(self.update_merged_view)

        slice_layout.addWidget(self.slice_label)
        slice_layout.addWidget(self.slice_slider)
        slice_layout.addWidget(self.slice_spinbox)

        visual_layout.addLayout(slice_layout)

        merged_layout.addLayout(visual_layout, stretch=2)

        # Controls area
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMaximumWidth(300)

        # Grid settings
        grid_group = QtWidgets.QGroupBox('Grid Settings')
        grid_layout = QtWidgets.QFormLayout()

        self.grid_size = QtWidgets.QSpinBox()
        self.grid_size.setMinimum(1)
        self.grid_size.setMaximum(5)
        self.grid_size.setValue(1)
        self.grid_size.valueChanged.connect(self.update_merged_view)
        grid_layout.addRow('Grid Size:', self.grid_size)

        # Grid overlay opacity
        self.grid_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grid_opacity.setMinimum(0)
        self.grid_opacity.setMaximum(100)
        self.grid_opacity.setValue(50)
        self.grid_opacity.valueChanged.connect(self.update_grid_opacity)
        grid_layout.addRow('Grid Opacity:', self.grid_opacity)

        grid_group.setLayout(grid_layout)
        controls_layout.addWidget(grid_group)

        # Display options
        display_group = QtWidgets.QGroupBox('Display Options')
        display_layout = QtWidgets.QVBoxLayout()

        # Add view mode selector
        display_layout.addWidget(QtWidgets.QLabel('View Mode:'))
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItems(['XY', 'XZ', 'YZ'])
        self.view_mode_combo.currentTextChanged.connect(self.change_view_mode)
        display_layout.addWidget(self.view_mode_combo)

        self.display_mean = QtWidgets.QRadioButton('Mean')
        self.display_median = QtWidgets.QRadioButton('Median')
        self.display_std = QtWidgets.QRadioButton('Standard Deviation')
        self.display_single = QtWidgets.QRadioButton('Single ROI')
        self.display_mean.setChecked(True)

        for radio in [
            self.display_mean,
            self.display_median,
            self.display_std,
            self.display_single,
        ]:
            display_layout.addWidget(radio)
            radio.toggled.connect(self.update_merged_view)

        # Add Single ROI selection
        single_roi_layout = QtWidgets.QHBoxLayout()
        self.single_roi_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.single_roi_spinbox = QtWidgets.QSpinBox()

        self.single_roi_slider.setMinimum(0)
        self.single_roi_spinbox.setMinimum(0)
        self.single_roi_spinbox.setMinimumWidth(75)

        # Initial maximum will be set when updating the view
        self.single_roi_slider.setMaximum(0)
        self.single_roi_spinbox.setMaximum(0)

        self.single_roi_slider.valueChanged.connect(self.single_roi_spinbox.setValue)
        self.single_roi_spinbox.valueChanged.connect(self.single_roi_slider.setValue)
        self.single_roi_slider.valueChanged.connect(self.update_merged_view)

        single_roi_layout.addWidget(QtWidgets.QLabel('ROI:'))
        single_roi_layout.addWidget(self.single_roi_slider)
        single_roi_layout.addWidget(self.single_roi_spinbox)

        # Create a widget to contain the ROI selection controls
        roi_selection_widget = QtWidgets.QWidget()
        roi_selection_widget.setLayout(single_roi_layout)
        roi_selection_widget.setEnabled(False)  # Initially disabled

        # Store reference to enable/disable later
        self.roi_selection_widget = roi_selection_widget

        # Connect radio button to enable/disable ROI selection
        self.display_single.toggled.connect(roi_selection_widget.setEnabled)

        display_layout.addWidget(roi_selection_widget)
        display_group.setLayout(display_layout)
        controls_layout.addWidget(display_group)

        # Auto level control
        auto_level_group = QtWidgets.QGroupBox('Level Control')
        auto_level_layout = QtWidgets.QVBoxLayout()
        self.auto_level_checkbox = QtWidgets.QCheckBox('Auto Level')
        self.auto_level_checkbox.setChecked(True)
        self.auto_level_checkbox.stateChanged.connect(self.update_merged_view)
        self.normalization_checkbox = QtWidgets.QCheckBox('Normalize (Field)')
        self.normalization_checkbox.setChecked(True)
        self.normalization_checkbox.stateChanged.connect(self.update_merged_view)
        auto_level_layout.addWidget(self.auto_level_checkbox)
        auto_level_layout.addWidget(self.normalization_checkbox)
        auto_level_group.setLayout(auto_level_layout)
        controls_layout.addWidget(auto_level_group)

        # Metadata display
        metadata_group = QtWidgets.QGroupBox('Metadata')
        metadata_layout = QtWidgets.QVBoxLayout()
        self.metadata_text = QtWidgets.QTextEdit()
        self.metadata_text.setReadOnly(True)
        metadata_layout.addWidget(self.metadata_text)
        metadata_group.setLayout(metadata_layout)
        controls_layout.addWidget(metadata_group)

        # Refresh button
        self.refresh_btn = QtWidgets.QPushButton('Refresh View')
        self.refresh_btn.clicked.connect(self.update_merged_view)
        self.zero_btn = QtWidgets.QPushButton(f'Zero Plane {self.psf_data.zero_plane}')
        self.zero_btn.clicked.connect(self.update_slice_controls)
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(self.zero_btn)

        # Export PSF data
        self.export_btn = QtWidgets.QPushButton('Export PSF Data')
        self.export_btn.clicked.connect(self._export_psf_data)
        controls_layout.addWidget(self.export_btn)

        # Add stretch to push controls to the top
        controls_layout.addStretch()

        merged_layout.addWidget(controls_widget)
        self.view_tabs.addTab(merged_widget, 'PSF View')

    def setup_statistics_tab(self):
        '''
        Set up the Statistics tab with separate left and right graphs and controls.
        '''
        stats_widget = QtWidgets.QWidget()
        stats_layout = QtWidgets.QHBoxLayout()
        stats_widget.setLayout(stats_layout)

        # Left graph (ROI plot)
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_widget.setLayout(left_layout)

        self.roi_plot = pg.PlotWidget(title='Statistics Over Z')
        self.roi_plot.setLabel('left', 'Value')
        self.roi_plot.setLabel('bottom', 'Z-Slice')
        left_layout.addWidget(self.roi_plot)

        # Add range ROI to the plot
        self.range_roi = pg.LinearRegionItem()
        self.range_roi.setZValue(10)
        self.roi_plot.addItem(self.range_roi)
        self.range_roi.sigRegionChanged.connect(self.update_range_spinboxes)

        left_options = QtWidgets.QFormLayout()
        left_layout.addLayout(left_options)

        # Statistic selection for ROI plot
        self.stat_combo = QtWidgets.QComboBox()
        for stat in self.psf_data.available_stats:
            self.stat_combo.addItem(stat)
        self.stat_combo.currentTextChanged.connect(self.update_roi_statistics)
        left_options.addRow(
            'Select statistic:',
            self.stat_combo,
        )
        # add confidence enum combo:
        self.confidence_combo = QtWidgets.QComboBox()
        for c in ConfidenceMethod:
            self.confidence_combo.addItem(c.name, c)
        self.confidence_combo.currentIndexChanged.connect(self.update_roi_statistics)
        left_options.addRow(
            'Confidence method:',
            self.confidence_combo,
        )
        # add confedence interval spin box 0 to 100
        self.confidence_spinbox = QtWidgets.QSpinBox()
        self.confidence_spinbox.setMinimum(0)
        self.confidence_spinbox.setMaximum(100)
        self.confidence_spinbox.setValue(95)
        self.confidence_spinbox.valueChanged.connect(self.update_roi_statistics)
        left_options.addRow(
            'Confidence interval:',
            self.confidence_spinbox,
        )

        # Refresh button for left graph
        self.refresh_roi_btn = QtWidgets.QPushButton('Refresh ROI Statistics')
        self.refresh_roi_btn.clicked.connect(self.update_roi_statistics)
        left_options.addWidget(self.refresh_roi_btn)

        # Add zero plane adjustment controls
        self.zero_plane_method = QtWidgets.QComboBox()
        self.zero_plane_method.addItems(
            ['Manual', 'Peak', 'Valley', 'Gaussian Fit', 'Gaussian Fit (Inverted)']
        )
        left_options.addRow('Zero Plane Method:', self.zero_plane_method)

        # Add range widgets
        range_controls_layout = QtWidgets.QHBoxLayout()

        # Update spinboxes for range selection
        self.z_range_start = QtWidgets.QSpinBox()
        self.z_range_start.setMinimumWidth(75)
        self.z_range_end = QtWidgets.QSpinBox()
        self.z_range_end.setMinimumWidth(75)
        range_controls_layout.addWidget(QtWidgets.QLabel('Range:'))
        range_controls_layout.addWidget(self.z_range_start)
        range_controls_layout.addWidget(QtWidgets.QLabel('to'))
        range_controls_layout.addWidget(self.z_range_end)

        self.z_range_start.valueChanged.connect(self.update_range_roi)
        self.z_range_end.valueChanged.connect(self.update_range_roi)
        # Add range controls to left options
        left_options.addRow('Range:', range_controls_layout)

        # Add CurveFitMethods Combo
        self.curve_fit_combo = QtWidgets.QComboBox()
        for m in stats.CurveFitMethod:
            self.curve_fit_combo.addItem(m.name, m)

        left_options.addRow('Curve Fit Method:', self.curve_fit_combo)

        # Add restore button
        buttons_layout = QtWidgets.QHBoxLayout()
        self.restore_zero_plane_btn = QtWidgets.QPushButton(
            'Restore Zero Plane', clicked=self.restore_zero_plane
        )
        buttons_layout.addWidget(self.restore_zero_plane_btn)

        self.adjust_zero_plane_btn = QtWidgets.QPushButton(
            'Adjust Zero Plane', clicked=self.adjust_zero_plane
        )
        buttons_layout.addWidget(self.adjust_zero_plane_btn)

        self.slope_btn = QtWidgets.QPushButton('Fit Curve', clicked=self.fit_stat_curve)
        buttons_layout.addWidget(self.slope_btn)
        self.export_fit_btn = QtWidgets.QPushButton(
            'Export', clicked=self.export_fit_curve
        )
        buttons_layout.addWidget(self.export_fit_btn)
        self.import_fit_btn = QtWidgets.QPushButton(
            'Import', clicked=self.import_fit_curve
        )
        buttons_layout.addWidget(self.import_fit_btn)

        left_options.addRow(buttons_layout)

        # Right graph (Intensity plot)
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.intensity_plot = pg.PlotWidget(title='Intensity Statistics Over Z')
        self.intensity_plot.setLabel('left', 'Intensity')
        self.intensity_plot.setLabel('bottom', 'Z-Slice')
        right_layout.addWidget(self.intensity_plot)

        # Refresh button for right graph
        self.refresh_intensity_btn = QtWidgets.QPushButton(
            'Refresh Intensity Statistics'
        )
        self.refresh_intensity_btn.clicked.connect(self.update_intensity_statistics)
        right_layout.addWidget(self.refresh_intensity_btn)

        stats_layout.addWidget(left_widget, 2)
        stats_layout.addWidget(right_widget, 1)

        self.view_tabs.addTab(stats_widget, 'Statistics')

    def setup_3d_view_tab(self):
        '''Set up the 3D View tab with enhanced controls.'''
        view_3d_widget = QtWidgets.QWidget()
        layout_3d = QtWidgets.QHBoxLayout()
        view_3d_widget.setLayout(layout_3d)

        # 3D visualization area
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=180, elevation=0, azimuth=90)
        layout_3d.addWidget(self.gl_widget, stretch=2)

        # Controls for 3D visualization
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMaximumWidth(300)

        # View controls
        view_controls = QtWidgets.QGroupBox('View Controls')
        view_layout = QtWidgets.QVBoxLayout()

        view_buttons = {
            'Top': (90, 0),
            'Bottom': (-90, 0),
            'Front': (0, 0),
            'Back': (0, 180),
            'Left': (0, 90),
            'Right': (0, -90),
        }

        for label, rotation in view_buttons.items():
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda checked, r=rotation: self.set_3d_view(r))
            view_layout.addWidget(btn)

        view_controls.setLayout(view_layout)
        controls_layout.addWidget(view_controls)

        # Data type selection
        type_group = QtWidgets.QGroupBox('Display Type')
        type_layout = QtWidgets.QVBoxLayout()
        self.view_3d_type = QtWidgets.QComboBox()
        display_types = [
            {
                'name': 'Mean',
                'value': 'mean',
            },
            {
                'name': 'Median',
                'value': 'median',
            },
            {
                'name': 'Standard Deviation',
                'value': 'std',
            },
        ]
        for t in display_types:
            self.view_3d_type.addItem(t['name'], t['value'])
        self.view_3d_type.currentTextChanged.connect(self.update_3d_view)
        type_layout.addWidget(self.view_3d_type)
        type_group.setLayout(type_layout)
        controls_layout.addWidget(type_group)

        # Opacity control
        opacity_group = QtWidgets.QGroupBox('Opacity')
        opacity_layout = QtWidgets.QVBoxLayout()
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(10)
        self.opacity_slider.valueChanged.connect(self.update_3d_view)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_group.setLayout(opacity_layout)
        controls_layout.addWidget(opacity_group)

        # Colormap preview
        colormap_group = QtWidgets.QGroupBox('Colormap')
        colormap_layout = QtWidgets.QVBoxLayout()
        self.colormap_preview = pg.GradientWidget(orientation='bottom')
        self.colormap_preview.setMinimumHeight(40)
        self.colormap_preview.item.setColorMap(pg.colormap.get('jet', 'matplotlib'))
        colormap_layout.addWidget(self.colormap_preview)
        colormap_group.setLayout(colormap_layout)
        controls_layout.addWidget(colormap_group)

        # Add axis labels checkbox
        self.show_labels = QtWidgets.QCheckBox('Show Axis Labels')
        self.show_labels.setChecked(True)
        self.show_labels.stateChanged.connect(self.update_3d_view)
        controls_layout.addWidget(self.show_labels)

        layout_3d.addWidget(controls_widget)
        self.view_tabs.addTab(view_3d_widget, '3D View')

    def set_3d_view(self, args: tuple[int]):
        '''Set the 3D view to a specific angle.'''
        self.gl_widget.opts['elevation'] = args[0]
        self.gl_widget.opts['azimuth'] = args[1]
        self.gl_widget.opts['distance'] = 180
        self.gl_widget.update()

    def get_display_type(self) -> str:
        '''Get the display type for the merged view.'''
        if self.display_mean.isChecked():
            return 'mean'
        elif self.display_median.isChecked():
            return 'median'
        elif self.display_std.isChecked():
            return 'std'
        else:
            return 'roi'

    def update_merged_view(self):
        if self.psf_data is None:
            return

        slice_index = self.slice_slider.value()
        grid_size = self.grid_size.value()
        # get display type (mean, median ... etc)
        display_type = self.get_display_type()
        norm = self.normalization_checkbox.isChecked()

        if self.view_mode == 'XY':
            roi_index = self.single_roi_slider.value()

            psf_image, grid_overlay = self.psf_data.get_z_slice(
                slice_index, display_type, roi_index, grid_size, norm
            )
        elif self.view_mode == 'XZ':
            psf_image, grid_overlay = self.psf_data.get_y_slice(
                slice_index, display_type, grid_size
            )
        else:  # YZ
            psf_image, grid_overlay = self.psf_data.get_x_slice(
                slice_index, display_type, grid_size
            )

        # Update the visualization
        if self.auto_level_checkbox.isChecked():
            self.psf_image_item.setImage(psf_image, autoLevels=True)
        else:
            self.psf_image_item.setImage(psf_image, autoLevels=False)

        if grid_overlay is not None:
            self.grid_overlay_item.setImage(grid_overlay)
            self.update_grid_opacity()
        else:
            self.grid_overlay_item.clear()

        # Update metadata
        self.update_metadata()

    def change_view_mode(self, mode):
        self.view_mode = mode
        self.update_slice_controls()
        self.update_merged_view()

    def update_metadata(self):
        '''Update the metadata display in the PSF view tab.'''
        if self.psf_data is None:
            return

        total_rois = sum([z['count'] for z in self.psf_data.zslices])
        max_rois_per_slice = max([z['count'] for z in self.psf_data.zslices])
        min_rois_per_slice = min([z['count'] for z in self.psf_data.zslices])
        metadata = f'''
        Total ROIs: {total_rois}
        Total Z-slices: {len(self.psf_data)}
        Average ROIs per slice: {total_rois / len(self.psf_data):.2f}
        Maximum ROIs in a slice: {max_rois_per_slice}
        Minimum ROIs in a slice: {min_rois_per_slice}
        Shape: {self.psf_data.shape}
        Pixel Size: {self.psf_data.pixel_size}
        Z Step Size: {self.psf_data.z_step} nm
        ROI Size: {self.psf_data.roi_size}
        Upsample: {self.psf_data.upsample}
        Stack: {self.psf_data.path}
        ROI Info: {self.psf_data.roi_info}
        Fit Method: {FittingMethod(self.psf_data.fitting_method)}
        Zero Plane: {self.psf_data.zero_plane}
        '''
        self.metadata_text.setText(metadata)

    def update_grid_opacity(self):
        '''Update the opacity of the grid overlay.'''
        self.grid_overlay_item.setOpacity(self.grid_opacity.value() / 100)

    def update_3d_view(self):
        '''Update the 3D visualization with enhanced features.'''
        self.gl_widget.clear()

        try:
            # Add coordinate axes with labels if enabled
            if self.show_labels.isChecked():
                ax = gl.GLAxisItem()
                ax.setSize(x=10, y=10, z=10)
                self.gl_widget.addItem(ax)

                # Add grid
                g = gl.GLGridItem()
                g.scale(10, 10, 1)
                self.gl_widget.addItem(g)

            zero = self.psf_data.zero_plane

            # Prepare volumetric data based on selected type
            view_type = self.view_3d_type.currentData()
            volume_data = self.psf_data.get_volume(view_type)

            # Normalize data
            if np.all(volume_data == 0):
                return

            normalized_data = (volume_data - volume_data.min()) / (
                volume_data.max() - volume_data.min()
            )

            # Create RGBA volume with colormap
            d2 = np.empty(volume_data.shape + (4,), dtype=np.ubyte)
            jet_colors = self.colormap_preview.item.colorMap().getLookupTable(
                0.0, 1.0, 256
            )
            color_indices = (normalized_data * 255).astype(np.uint8)

            # Set RGB channels using jet colormap
            d2[..., 0] = jet_colors[color_indices, 0]  # Red channel
            d2[..., 1] = jet_colors[color_indices, 1]  # Green channel
            d2[..., 2] = jet_colors[color_indices, 2]  # Blue channel

            # Set alpha channel based on opacity slider and data intensity
            opacity = self.opacity_slider.value() / 100.0
            d2[..., 3] = (normalized_data * 255 * opacity).astype(np.ubyte)

            # Set scaling ratios for X and Y axes
            x_ratio = y_ratio = 1  # X and Y are the reference, Z scaled
            # z_ratio
            z_ratio = self.psf_data.get_ratio()

            # Create and add volume item
            v = gl.GLVolumeItem(d2)

            v.scale(z_ratio, y_ratio, x_ratio)  # Apply scaling in (z, y, x) order

            v.translate(
                -zero * z_ratio,
                -volume_data.shape[1] // 2 * y_ratio,
                -volume_data.shape[2] // 2 * x_ratio,
            )
            self.gl_widget.addItem(v)

            # Add tick labels if enabled
            if self.show_labels.isChecked():
                for i in range(volume_data.shape[0]):
                    if i % 5 == 0:  # Add label every 5 slices
                        text = gl.GLTextItem(
                            pos=np.array(
                                [
                                    -volume_data.shape[0] // 2,
                                    -volume_data.shape[1] // 2,
                                    i - volume_data.shape[2] // 2,
                                ]
                            ),
                            text=str(i),
                            color=(1, 1, 1, 1),
                        )
                        self.gl_widget.addItem(text)

        except Exception as e:
            print(f'Error updating 3D view: {str(e)}')

    def update_range_spinboxes(self):
        mn, mx = self.range_roi.getRegion()
        zero = self.psf_data.zero_plane
        self.z_range_start.setValue(int(mn / self.psf_data.z_step + zero))
        self.z_range_end.setValue(int(mx / self.psf_data.z_step + zero))

    def update_range_roi(self):
        zero = self.psf_data.zero_plane
        start = (self.z_range_start.value() - zero) * self.psf_data.z_step
        end = (self.z_range_end.value() - zero) * self.psf_data.z_step
        self.range_roi.setRegion((start, end))

    def restore_zero_plane(self):
        if hasattr(self, 'old_zero_plane'):
            self.psf_data.zero_plane = self.psf_data.old_zero_plane
            self.update_all_views()

    def adjust_zero_plane(self):
        if self.psf_data is None:
            return

        method = self.zero_plane_method.currentText()
        selected_stat = self.stat_combo.currentText()

        self.psf_data.adjust_zero_plane(
            selected_stat, method, self.range_roi.getRegion()
        )

        # Update the plots
        self.update_roi_statistics()
        self.update_intensity_statistics()

    def fit_stat_curve(self):
        if self.psf_data is None:
            return

        selected_stat = self.stat_combo.currentText()
        self.fit_result = self.psf_data.get_z_cal(
            selected_stat=selected_stat,
            region=self.range_roi.getRegion(),
            confidence_method=self.confidence_combo.currentData(),
            confidence_level=self.confidence_spinbox.value() / 100,
            method=self.curve_fit_combo.currentData(),
        )

        self.plot_curve_fit()

    def plot_curve_fit(self):
        if self.fit_result:
            # Plot the curve fit
            if isinstance(self.fit_result, stats.SlopeResult):
                x_fit = (
                    np.arange(len(self.psf_data)) - self.psf_data.zero_plane
                ) * self.psf_data.z_step
                y_fit = self.fit_result.slope * x_fit + self.fit_result.intercept

                y_lower = y_upper = None
                if self.fit_result.slope_ci:
                    y_lower = (
                        self.fit_result.slope_ci[0] * x_fit + self.fit_result.intercept
                    )
                    y_upper = (
                        self.fit_result.slope_ci[1] * x_fit + self.fit_result.intercept
                    )

                self._add_stat_with_confidence(
                    x_fit,
                    y_fit,
                    y_lower,
                    y_upper,
                    'Calibration Slope',
                    (0, 255, 0, 150),
                )
            elif isinstance(self.fit_result, stats.CurveResult):
                x_fit = (
                    np.arange(len(self.psf_data)) - self.psf_data.zero_plane
                ) * self.psf_data.z_step
                y_fit = self.fit_result.get_data(x_fit)

                y_lower = y_upper = None

                self._add_stat_with_confidence(
                    x_fit,
                    y_fit,
                    y_lower,
                    y_upper,
                    'Calibration Curve',
                    (0, 255, 0, 150),
                )
            elif isinstance(self.fit_result, dict):
                x_fit = self.fit_result['fitted_curves']['z']
                y_fit = [
                    self.fit_result['fitted_curves']['sigma_x'],
                    self.fit_result['fitted_curves']['sigma_y'],
                ]

                y_lower = y_upper = None

                self._add_stat_with_confidence(
                    x_fit,
                    y_fit[0],
                    y_lower,
                    y_upper,
                    'Sigma X Curve',
                    (0, 255, 0, 150),
                )
                self._add_stat_with_confidence(
                    x_fit,
                    y_fit[1],
                    y_lower,
                    y_upper,
                    'Sigma X Curve',
                    (0, 255, 255, 150),
                )

            # use tabulate to print out the slope results info in stats text
            # pprint(asdict(self.fit_result))

    def export_fit_curve(self):
        if self.fit_result is None:
            return

        if stats.export_fit_curve(
            self.fit_result, self, os.path.dirname(self.psf_data.path)
        ):
            print(
                f'Fit curve exported!'
            )

    def import_fit_curve(self):
        fit_result, _ = stats.import_fit_curve(
            self,
            os.path.dirname(self.psf_data.path),
        )

        if fit_result:
            self.fit_result = fit_result
            self.range_roi.setRegion(self.fit_result.significant_region)
            self.plot_curve_fit()

    def update_intensity_statistics(self):
        '''Update intensity statistics plot.'''
        if self.psf_data is None:
            return

        self.intensity_plot.clear()

        _indices, _mean, _median, _std = self.psf_data.get_intensity_stats()

        # Plot intensity statistics
        self.intensity_plot.plot(_indices, _mean, pen='r', name='Mean')
        self.intensity_plot.plot(_indices, _median, pen='g', name='Median')
        self.intensity_plot.plot(_indices, _std, pen='y', name='Std Dev')

        # Update zero plane line
        zero_line_intensity = pg.InfiniteLine(pos=0, angle=90, pen='w', movable=False)
        self.intensity_plot.addItem(zero_line_intensity)

        # Add legend
        self.intensity_plot.addLegend()

        # Set plot title and labels
        self.intensity_plot.setTitle('Intensity Statistics Over Z')
        self.intensity_plot.setLabel('left', 'Intensity')
        self.intensity_plot.setLabel('bottom', 'Z-Slice')

    def update_all_views(self, extra=True):
        '''Update all views when data changes.'''
        self.update_merged_view()
        self.update_metadata()
        self.update_3d_view()
        if extra:
            self.update_roi_statistics()
            self.update_intensity_statistics()

    def update_slice_controls(self):
        if self.view_mode == 'XY':
            self.slice_label.setText('Z-Slice:')
            max_slice = len(self.psf_data) - 1

            # set viewbox aspect ratio
            ratio = 1
            zero = self.psf_data.zero_plane
        else:
            if self.view_mode == 'XZ':
                self.slice_label.setText('Y-Slice:')
            else:  # YZ
                self.slice_label.setText('X-Slice:')

            max_slice = self.psf_data.roi_size - 1

            # set viewbox aspect ratio
            ratio = self.psf_data.get_ratio()
            zero = max_slice // 2

        # set viewbox aspect ratio
        self.view_box.setAspectLocked(True, ratio)

        self.slice_slider.setMaximum(max_slice)
        self.slice_spinbox.setMaximum(max_slice)
        self.zero_btn.setText(f'Zero Plane {zero}')
        self.slice_slider.setValue(zero)

    def set_psf_data(self, psf_data: 'PSFdata'):
        '''Set new PSF data and update all views.

        Parameters
        ----------
        psf_data : PSFdata
            New PSF data to visualize
        '''
        self.psf_data = psf_data

        # Update slice controls
        self.update_slice_controls()
        # Update all views
        self.update_all_views()

    def update_roi_statistics(self):
        '''Update ROI statistics plot with confidence intervals where applicable.'''
        if self.psf_data is None:
            return

        self.roi_plot.clear()
        selected_stat = self.stat_combo.currentText()
        z_indices, param_stat, param_min, param_max = self.psf_data.get_stats(
            selected_stat,
            self.confidence_combo.currentData(),
            self.confidence_spinbox.value() / 100,
        )

        # Plot based on statistic type
        if selected_stat == 'Sigma':
            self._plot_sigma_stats(z_indices, param_stat, param_min, param_max)
        else:
            self._plot_single_stat(
                z_indices, param_stat, param_min, param_max, selected_stat
            )

        self._setup_plot_elements(z_indices, selected_stat)

    def _plot_sigma_stats(self, z_indices, param_stat, param_min, param_max):
        '''Plot sigma statistics with confidence intervals.'''
        colors = {
            'x': (0, 0, 255, 150),
            'y': (255, 0, 0, 150),
        }  # Semi-transparent blue and red

        if 1 <= len(param_stat) <= 2:
            # Plot X sigma
            self._add_stat_with_confidence(
                z_indices,
                param_stat[0],
                param_min[0],
                param_max[0],
                'Sigma X',
                colors['x'],
            )

            # Plot Y sigma if available
            if len(param_stat) == 2:
                self._add_stat_with_confidence(
                    z_indices,
                    param_stat[1],
                    param_min[1],
                    param_max[1],
                    'Sigma Y',
                    colors['y'],
                )

    def _plot_single_stat(self, z_indices, param_stat, param_min, param_max, stat_name):
        '''Plot a single statistic with confidence interval.'''
        self._add_stat_with_confidence(
            z_indices,
            param_stat,
            param_min,
            param_max,
            stat_name,
            (0, 0, 255, 150),  # Semi-transparent blue
        )

    def _add_stat_with_confidence(self, x, y, y_min, y_max, name, color):
        '''Add a statistic line with confidence interval fill.'''
        # Main line
        pen = pg.mkPen(color=color, width=2)
        self.roi_plot.plot(x, y, pen=pen, name=name)

        # Confidence interval fill
        if y_min is not None and y_max is not None:
            fill = pg.FillBetweenItem(
                pg.PlotCurveItem(x, y_max),
                pg.PlotCurveItem(x, y_min),
                brush=pg.mkBrush((*color[:3], 50)),  # More transparent for fill
            )
            self.roi_plot.addItem(fill)

    def _setup_plot_elements(self, z_indices, selected_stat):
        '''Setup common plot elements.'''
        # Zero plane line
        zero_line_roi = pg.InfiniteLine(pos=0, angle=90, pen='w', movable=False)
        self.roi_plot.addItem(zero_line_roi)

        # Range ROI setup
        z_range = (0, (len(self.psf_data) - 1))
        self.range_roi.setRegion((z_indices.min(), z_indices.max()))

        # Update range controls
        self._update_range_controls(z_range)

        # Add plot elements
        self.roi_plot.addItem(self.range_roi)
        self.roi_plot.addLegend()

        # Set labels
        self._set_plot_labels(selected_stat)

    def _update_range_controls(self, z_range):
        '''Update range control values and limits.'''
        self.z_range_start.setRange(*z_range)
        self.z_range_end.setRange(*z_range)
        self.z_range_start.setValue(0)
        self.z_range_end.setValue(z_range[1])

    def _set_plot_labels(self, selected_stat):
        '''Set plot title and axis labels.'''
        self.roi_plot.setTitle(f'{selected_stat} Over Z')
        self.roi_plot.setLabel('left', selected_stat)
        self.roi_plot.setLabel('bottom', 'Z-Slice')

    def _export_psf_data(self):
        '''Export PSF data to an HDF5 file.'''
        filename, _ = getSaveFileName(
            self,
            'Save PSF Data',
            os.path.dirname(self.psf_data.path),
            'HDF5 PSF Files (*.psf.h5)',
        )

        if filename:
            self.psf_data.save_hdf(filename)
