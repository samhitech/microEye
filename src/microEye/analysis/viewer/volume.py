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
    '''Localizations in 3D, each drawn as a small sprite of fixed screen size.'''

    #: (elevation, azimuth) of the view buttons
    VIEWS = {'Tilted': (25, -60), 'Top': (90, -90), 'Front': (0, -90), 'Side': (0, 0)}

    def __init__(self, x, y, z, intensity=None, title='3D view'):
        '''
        Parameters
        ----------
        x, y, z : np.ndarray
            finite coordinates in nm
        intensity : np.ndarray, optional
            enables colouring by intensity
        title : str, optional
            the window title
        '''
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 800)

        xyz = np.c_[x, y, z].astype(np.float32)
        self.xyz = xyz - (xyz.min(axis=0) + xyz.max(axis=0)) / 2  # centred
        self.intensity = intensity
        span = np.ptp(self.xyz, axis=0)
        self.span = float(max(span[0], span[1], 1.0))

        self.view = gl.GLViewWidget()
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=self.span, y=self.span)
        self.grid.setSpacing(x=self.span / 10, y=self.span / 10)
        self.view.addItem(self.grid)
        # additive: overlapping sprites add up, so dense regions are brighter
        self.scatter = gl.GLScatterPlotItem(pxMode=True, glOptions='additive')
        self.view.addItem(self.scatter)

        self.color_by = QtWidgets.QComboBox()
        self.color_by.addItems(['z'] if intensity is None else ['z', 'intensity'])
        self.colormap = pg.GradientWidget(orientation='bottom')
        self.colormap.item.setColorMap(pg.colormap.get('CET-R4'))
        self.point_size = QtWidgets.QSpinBox()
        self.point_size.setRange(1, 20)
        self.point_size.setValue(3)
        self.point_size.setSuffix(' px')
        self.opacity = QtWidgets.QDoubleSpinBox()
        self.opacity.setRange(0.05, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.5)
        self.z_scale = QtWidgets.QDoubleSpinBox()
        self.z_scale.setRange(1.0, 50.0)
        self.z_scale.setSuffix(' ×')
        self.z_scale.setToolTip('Stretches z so a thin layer shows its depth.')
        # a depth of a quarter of the width, at most 20 times stretched
        self.z_scale.setValue(
            float(np.clip(round(0.25 * self.span / max(span[2], 1.0)), 1, 20))
        )
        show_grid = QtWidgets.QCheckBox('Grid')
        show_grid.setChecked(True)

        views = QtWidgets.QHBoxLayout()
        for name, angles in self.VIEWS.items():
            button = QtWidgets.QPushButton(name)
            button.clicked.connect(lambda _=False, a=angles: self.look(*a))
            views.addWidget(button)

        form = QtWidgets.QFormLayout()
        form.addRow('Colour by', self.color_by)
        form.addRow(self.colormap)
        form.addRow('Point size', self.point_size)
        form.addRow('Opacity', self.opacity)
        form.addRow('Z stretch', self.z_scale)
        form.addRow(show_grid)
        form.addRow(views)
        panel = QtWidgets.QWidget()
        panel.setLayout(form)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.view, stretch=4)
        layout.addWidget(panel, stretch=1)

        self.color_by.currentIndexChanged.connect(self.update_colors)
        self.colormap.sigGradientChanged.connect(self.update_colors)
        self.opacity.valueChanged.connect(self.update_colors)
        self.point_size.valueChanged.connect(
            lambda size: self.scatter.setData(size=size)
        )
        self.z_scale.valueChanged.connect(self.update_positions)
        show_grid.toggled.connect(self.grid.setVisible)

        self.scatter.setData(size=self.point_size.value())
        self.update_positions()
        self.update_colors()
        self.look(*self.VIEWS['Tilted'])

    def update_positions(self):
        pos = self.xyz * np.float32([1, 1, self.z_scale.value()])
        self.scatter.setData(pos=pos)
        self.grid.resetTransform()
        self.grid.translate(0, 0, float(pos[:, 2].min()))

    def update_colors(self):
        by_z = self.color_by.currentText() == 'z'
        values = self.xyz[:, 2] if by_z else np.asarray(self.intensity, np.float32)
        low, high = np.nanpercentile(values, [1, 99])
        scaled = np.clip((values - low) / ((high - low) or 1.0), 0, 1)
        colors = self.colormap.item.colorMap().map(scaled, mode='float')
        colors[:, 3] = self.opacity.value()
        self.scatter.setData(color=colors)

    def look(self, elevation, azimuth):
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(0, 0, 0),
            distance=1.5 * self.span,
            elevation=elevation,
            azimuth=azimuth,
        )



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
