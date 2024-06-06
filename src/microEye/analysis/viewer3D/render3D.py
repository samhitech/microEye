import sys

import numpy as np
from microEye.qt import QApplication, QtWidgets
from vispy import app, color, scene


class VoxelViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.voxels = np.random.randint(0, 2, size=(10, 10, 10))

        # Create Vispy canvas
        self.canvas = scene.SceneCanvas(keys='interactive')
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        # Create volume visual with 'jet' colormap
        cmap = color.get_colormap('jet')
        self.volume = scene.visuals.Volume(self.voxels, cmap=cmap, method='minip')

        # Create view
        self.view = self.canvas.central_widget.add_view()
        self.view.add(self.volume)

        # Set camera
        self.view.camera = 'arcball'
        self.view.camera.set_range()

        # Add XYZ axes with adjusted position
        axes = scene.visuals.XYZAxis(parent=self.view.scene)

        # Set up layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

        self.show()

    def on_draw(self, event):
        self.canvas.update()

def main():
    appQt = QApplication(sys.argv)
    viewer = VoxelViewer()
    sys.exit(appQt.exec())

if __name__ == '__main__':
    main()
