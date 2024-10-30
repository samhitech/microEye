import contextlib

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from microEye.analysis.rendering.cloud import normalize_intensities
from microEye.analysis.rendering.volumetric import (
    create_rgba_volume,
    normalize_volume,
)
from microEye.qt import Qt, QtGui, QtWidgets


class VolumeViewerWindow(QtWidgets.QWidget):
    def __init__(self, volume_data: np.ndarray = None, metadata=None):
        super().__init__()
        self.setWindowTitle('3D Volume Viewer')

        # Store data and metadata
        self.volume_data = volume_data
        self.normalized_data = None
        self.metadata = (
            metadata
            if metadata is not None
            else {
                'voxel_size': {'x': 1, 'y': 1, 'z': 1},
                'coordinates': {'z_min': 0, 'z_max': 1},
            }
        )

        # Create the main widget and layout
        self.setLayout(QtWidgets.QHBoxLayout())

        # Create 3D viewport
        self.init_3d_viewport()

        # Create control panel
        self.init_control_panel()

        # Set window size
        self.resize(1200, 800)

        # Initialize volume visualization if data is provided
        if volume_data is not None:
            self.set_volume_data(volume_data, metadata)

    def init_3d_viewport(self):
        '''Initialize the 3D viewport with PyQtGraph GLViewWidget.'''
        self.view = gl.GLViewWidget()
        self.layout().addWidget(self.view, stretch=4)

        self.axis = gl.GLAxisItem()
        self.axis.setSize(x=200, y=200, z=200)
        self.view.addItem(self.axis)

        self.grid = gl.GLGridItem()
        self.grid.setSize(x=200, y=200)
        self.grid.translate(dx=0, dy=0, dz=0)
        self.view.addItem(self.grid)

        self.volume_item = None

    def init_control_panel(self):
        '''Initialize the control panel with sliders and options.'''
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        self.layout().addWidget(control_widget, stretch=1)

        # Add color map gradient widget
        colormap_label = QtWidgets.QLabel('Color Map')
        control_layout.addWidget(colormap_label)

        self.colormap_preview = pg.GradientWidget(orientation='bottom')
        self.colormap_preview.setMinimumHeight(40)
        self.colormap_preview.item.setColorMap(pg.colormap.get('jet', 'matplotlib'))
        self.colormap_preview.sigGradientChanged.connect(self.update_volume)
        control_layout.addWidget(self.colormap_preview)

        # Add opacity slider
        opacity_label = QtWidgets.QLabel('Opacity')
        control_layout.addWidget(opacity_label)

        self.opacity_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.update_volume)
        control_layout.addWidget(self.opacity_slider)

        # Add spacer
        control_layout.addStretch()

    def update_volume(self):
        '''
        Update the volume visualization with current color map and opacity settings.
        Using accelerated processing functions.
        '''
        if self.volume_data is None:
            return

        # Remove existing volume if any
        if self.volume_item is not None:
            with contextlib.suppress(Exception):
                self.view.removeItem(self.volume_item)

        # Normalize data using accelerated function
        if self.normalized_data is None:
            self.normalized_data = normalize_volume(self.volume_data)

        # Get color lookup table from gradient widget
        colors = self.colormap_preview.item.colorMap().getLookupTable(0.0, 1.0, 256)
        colors = colors[:, :3].astype(np.uint8)  # Extract RGB components

        # Create RGBA volume with accelerated function
        opacity = self.opacity_slider.value() / 100.0

        rgba_volume = create_rgba_volume(self.normalized_data, colors, opacity)

        # Create volume item with proper scaling
        scale = (
            self.metadata['voxel_size']['x'],
            self.metadata['voxel_size']['y'],
            self.metadata['voxel_size']['z'],
        )

        self.volume_item = gl.GLVolumeItem(rgba_volume)

        # Position the volume with respect to Z zero point
        z_offset = self.metadata['coordinates']['z_min']
        self.volume_item.translate(0, 0, z_offset)

        # Add to view
        self.view.addItem(self.volume_item)

    def set_volume_data(self, volume_data, metadata=None):
        '''Set or update the volume data to visualize.'''
        self.volume_data = volume_data
        if metadata is not None:
            self.metadata = metadata

        # Update the visualization
        self.update_volume()


class PointCloudViewer(QtWidgets.QWidget):
    def __init__(
        self, points: np.ndarray = None, intensities: np.ndarray = None, metadata=None
    ):
        '''
        Initialize the point cloud viewer.

        Args:
            points: Nx3 numpy array of point coordinates (x, y, z)
            intensities: N numpy array of intensity values
            metadata: Dictionary containing additional information
        '''
        super().__init__()
        self.setWindowTitle('3D Point Cloud Viewer')

        # Store data and metadata
        self.points = points
        self.intensities = intensities
        self.metadata = (
            metadata
            if metadata is not None
            else {
                'point_size': 2,
                'coordinates': {'z_min': 0, 'z_max': 1},
                'bin_size': {
                    'x': 10,
                    'y': 10,
                    'z': 10,
                },
            }
        )

        # Create the main widget and layout
        self.main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.main_layout)

        # Create 3D viewport
        self.init_3d_viewport()

        # Create control panel
        self.init_control_panel()

        # Set window size
        self.resize(1200, 800)

        self.show()

        # Initialize point cloud visualization if data is provided
        # if points is not None:
        #     self.update_point_cloud()

        # Initialize histogram if we have data
        if self.intensities is not None:
            self.update_histogram()

    def set_3d_view(self, args: tuple[int]):
        '''Set the 3D view to a specific angle.'''
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(*(np.max(self.points, axis=0) / 2)),
            elevation=args[0],
            azimuth=args[1],
            distance=1.1 * np.max(self.points),
        )

    def closeEvent(self, event: QtGui.QCloseEvent):
        '''Cleanup and close the window.

        Params
        ------
        event : QCloseEvent
            Close event object
        '''
        self.view.clear()
        self.view.close()
        event.accept()

    def init_3d_viewport(self):
        '''Initialize the 3D viewport with PyQtGraph GLViewWidget.'''
        self.view = gl.GLViewWidget()
        self.main_layout.addWidget(self.view, stretch=4)

        # Set some reasonable defaults for camera
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(*np.mean(self.points, axis=0)),
            elevation=-90,
            azimuth=-90,
            distance=1.1 * np.max(self.points),
        )
        # make view ortho instead of perspective
        self.view.opts['fov'] = 60
        self.view.update()

    def init_control_panel(self):
        '''Initialize the control panel with point size and color options.'''
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        self.main_layout.addWidget(control_widget, stretch=1)

        # Add color map gradient widget
        colormap_group = QtWidgets.QGroupBox('Color Map')
        colormap_layout = QtWidgets.QVBoxLayout()
        colormap_group.setLayout(colormap_layout)

        self.colormap_preview = pg.GradientWidget(orientation='bottom')
        self.colormap_preview.setMinimumHeight(40)
        self.colormap_preview.item.setColorMap(pg.colormap.get('jet', 'matplotlib'))
        self.colormap_preview.sigGradientChanged.connect(self.update_point_cloud)
        colormap_layout.addWidget(self.colormap_preview)

        # Add preset buttons
        preset_layout = QtWidgets.QHBoxLayout()
        presets = ['viridis', 'plasma', 'inferno', 'magma']
        for preset in presets:
            btn = QtWidgets.QPushButton(preset)
            btn.setMaximumWidth(60)
            btn.clicked.connect(
                lambda checked, p=preset: self.colormap_preview.loadPreset(p)
            )
            preset_layout.addWidget(btn)
        colormap_layout.addLayout(preset_layout)
        control_layout.addWidget(colormap_group)

        # Add point size control
        display_options = QtWidgets.QGroupBox('Display Options')
        display_layout = QtWidgets.QFormLayout()
        display_options.setLayout(display_layout)

        self.size_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(100)
        self.size_slider.setValue(self.metadata['bin_size']['x'])
        self.size_slider.valueChanged.connect(self.update_point_cloud)

        size_display = QtWidgets.QSpinBox()
        size_display.setMinimum(1)
        size_display.setMaximum(100)
        size_display.setValue(self.size_slider.value())
        self.size_slider.valueChanged.connect(size_display.setValue)
        size_display.valueChanged.connect(self.size_slider.setValue)

        display_layout.addWidget(self.size_slider)
        display_layout.addWidget(size_display)

        # Add grid and axis to display
        self.axis_checkbox = QtWidgets.QCheckBox('Show Axis')
        self.axis_checkbox.setChecked(True)
        self.axis_checkbox.stateChanged.connect(lambda: self.update_point_cloud())
        display_layout.addWidget(self.axis_checkbox)
        self.grid_checkbox = QtWidgets.QCheckBox('Show Grid')
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(lambda: self.update_point_cloud())
        display_layout.addWidget(self.grid_checkbox)

        fov_spin = QtWidgets.QSpinBox()
        fov_spin.setMinimum(1)
        fov_spin.setMaximum(180)
        fov_spin.setValue(1)
        fov_spin.valueChanged.connect(
            lambda: self.view.setCameraParams(fov=fov_spin.value())
        )
        display_layout.addWidget(fov_spin)

        control_layout.addWidget(display_options)

        # View controls
        view_controls = QtWidgets.QGroupBox('View Controls')
        view_layout = QtWidgets.QHBoxLayout()

        view_buttons = {
            'T': (-90, -90),
            'B': (90, 90),
            'F': (0, 0),
            'Bck': (0, 180),
            'L': (0, 90),
            'R': (0, -90),
        }

        for label, rotation in view_buttons.items():
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda checked, r=rotation: self.set_3d_view(r))
            view_layout.addWidget(btn)

        view_controls.setLayout(view_layout)
        control_layout.addWidget(view_controls)

        # Add intensity histogram with range selection
        intensity_group = QtWidgets.QGroupBox('Intensity Range')
        intensity_layout = QtWidgets.QVBoxLayout()
        intensity_group.setLayout(intensity_layout)

        # Create histogram plot
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setMinimumHeight(100)
        intensity_layout.addWidget(self.hist_plot)

        # Reset range
        reset_button = QtWidgets.QPushButton('Reset Range')
        reset_button.clicked.connect(lambda: self.update_histogram())
        intensity_layout.addWidget(reset_button)

        # Create region selection
        self.region = pg.LinearRegionItem([0, 100])
        self.region.sigRegionChangeFinished.connect(self.update_point_cloud)
        self.hist_plot.addItem(self.region)

        control_layout.addWidget(intensity_group)

        # Add spacer
        control_layout.addStretch()

    def update_histogram(self):
        '''Update the intensity histogram display.'''
        if self.intensities is None:
            return

        # Clear previous histogram
        self.hist_plot.clear()

        # Calculate histogram
        y, x = np.histogram(self.intensities, bins=1000)

        # Create histogram
        self.hist_plot.plot(
            x, y, stepMode='center', fillLevel=0, brush=(0, 0, 255, 150)
        )

        # Add region selection
        if self.region not in self.hist_plot.items():
            self.hist_plot.addItem(self.region)

        # Set initial region positions to 1st and 99th percentiles
        p1, p99 = np.percentile(self.intensities, [1, 99])
        self.region.setRegion([p1, p99])

    def map_colors(self, intensities):
        '''Map intensity values to colors using current colormap.'''
        # Get intensity range from region selection
        lower, upper = self.region.getRegion()

        # Clip and normalize intensities
        norm_intensities = np.clip(intensities, lower, upper)
        norm_intensities = (norm_intensities - lower) / (upper - lower)

        # Get color lookup table from gradient widget
        colormap = self.colormap_preview.item.colorMap()
        colors = colormap.map(norm_intensities, mode='float')

        colors[intensities < lower, 3] = 0

        return colors

    def update_point_cloud(self):
        '''Update the point cloud visualization with current settings.'''
        if self.points is None:
            return

        # Remove existing scatter if any
        self.view.clear()

        # Map intensities to colors
        if self.intensities is not None:
            colors = self.map_colors(self.intensities)
        else:
            # Use height (z-coordinate) for coloring if no intensities provided
            z_values = self.points[:, 2]
            colors = self.map_colors(z_values)

        # Add axis and grid based on data extent
        data_range = np.ptp(self.points, axis=0)
        grid_size = max(data_range[0], data_range[1])

        if self.grid_checkbox.isChecked():
            grid = gl.GLGridItem(QtGui.QVector3D(grid_size, grid_size, 1))
            grid.translate(dx=grid_size / 2, dy=grid_size / 2, dz=0)
            grid.setSpacing(x=grid_size / 10, y=grid_size / 10)
            self.view.addItem(grid)

        if self.axis_checkbox.isChecked():
            axis = gl.GLAxisItem()
            axis.setSize(x=grid_size / 10, y=grid_size / 10, z=data_range[2] / 10)
            self.view.addItem(axis)

        # Create scatter plot item
        scatter_item = gl.GLScatterPlotItem(
            pos=self.points,
            color=colors,
            size=self.size_slider.value(),
            pxMode=False,
        )

        # Add to view
        self.view.addItem(scatter_item)


def show_volume(volume_data, metadata=None):
    '''Convenience function to show volume data in a new window.'''
    import sys

    from microEye.qt import QApplication

    # Create QApplication instance if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create and show window
    viewer = VolumeViewerWindow(volume_data, metadata)
    viewer.show()

    # Return viewer instance
    return viewer
