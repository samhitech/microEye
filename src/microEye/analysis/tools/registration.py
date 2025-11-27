import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from weakref import ref

import cv2
import numpy as np
import ome_types as ome
import tifffile as tf
from pystackreg import StackReg
from skimage import transform
from tqdm import tqdm

from microEye import __version__
from microEye.images import ImageSequenceBase, TiffSeqHandler, create_array
from microEye.qt import Qt, QtCore, QtGui, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.display import DisplayManager, fast_autolevels_opencv
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)


class RegistrationMethod(Enum):
    TRANSLATION = 2
    '''
    Translation
    '''

    RIGID_BODY = 3
    '''
    Rigid body (translation + rotation)
    '''

    SCALED_ROTATION = 4
    '''
    Scaled rotation (translation + rotation + scaling)
    '''

    AFFINE = 6
    '''
    Affine (translation + rotation + scaling + shearing)
    '''

    BILINEAR = 8
    '''
    Bilinear (non-linear transformation; does not preserve straight lines)
    '''

    @classmethod
    def from_string(cls, method_str: str):
        method_str = method_str.replace(' ', '_').upper()
        return cls[method_str] if method_str in cls.__members__ else None


def display_frame(
    name,
    frame: np.ndarray,
    histogram: bool = True,
    autoLevels: bool = True,
    stats: bool = False,
):
    thresholds, hist, cdf = fast_autolevels_opencv(frame, levels=autoLevels)

    # cv2.imshow(name, frame)
    DisplayManager.instance().image_update_signal.emit(
        name,
        frame,
        {
            'aspect_ratio': 1.0,
            'width': frame.shape[1],
            'height': frame.shape[0],
            'show_stats': stats,
            'autoLevels': autoLevels,
            'isSingleView': False,
            'isROIsView': False,
            'isOverlaidView': True,
            'plot_type': False,
            'threshold': thresholds,
            'plot': hist if histogram else cdf,
        },
    )


def cv2_warp(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    float_normalized = img.astype(np.float32) / np.iinfo(img.dtype).max
    transformed_img = cv2.warpPerspective(
        float_normalized,
        np.linalg.inv(matrix),
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    return (transformed_img * np.iinfo(img.dtype).max).astype(img.dtype)


def process_stack(
    filename: str,
    ref_stack: ImageSequenceBase | None,
    moving_stack: ImageSequenceBase,
    matrix: np.ndarray,
    ref_channel: int = 0,
    moving_channel: int = 0,
    event=None,
) -> bool:
    try:
        shape = moving_stack.shapeTCZYX()
        num_frames = shape[0]
        channels = 1

        if ref_stack is not None:
            if any(ref_stack.shapeTCZYX()[i] != shape[i] for i in [0, 2, 3, 4]):
                logger.error(
                    'Reference and moving stacks must have the same dimensions '
                    '(T, Z, Y, X).',
                )
                return False
            if ref_stack._dtype != moving_stack._dtype:
                logger.error(
                    'Reference and moving stacks must have the same data type.'
                )
                return False
            channels = 2

        shape = (num_frames, channels) + shape[2:]

        if filename.endswith('.zarr'):
            writer = create_array(
                filename=filename,
                shape=shape,
                chunks=(1, 1, 1, shape[-2], shape[-1]),
                dtype=moving_stack.getSlice(0, 0, 0).dtype,
                is_zip=True,
            )
        elif filename.endswith('.tif'):
            writer = tf.TiffWriter(
                filename,
                append=False,
                bigtiff=True,
                ome=False,
            )
        else:
            logger.error('Unsupported file format. Please select a .zarr or .tif file.')
            return False

        success = False

        def process_image(k: int):
            moving = moving_stack.getSlice(k, moving_channel).squeeze()

            data: np.ndarray = cv2_warp(moving, matrix)

            if channels == 2 and ref_stack is not None:
                reference = ref_stack.getSlice(k, ref_channel).squeeze()
                data = np.stack((reference, data), axis=0)

            if filename.endswith('.zarr'):
                writer[k, :, 0, ...] = data
            elif filename.endswith('.tif'):
                data = (
                    data[np.newaxis, np.newaxis, np.newaxis, ...]
                    if data.ndim == 2
                    else data[np.newaxis, :, np.newaxis, ...]
                )
                writer.write(
                    data=data.astype(np.uint16),
                    photometric='minisblack',
                )

        if filename.endswith('.tif'):
            for k in tqdm(range(num_frames), desc='Processing images'):
                process_image(k)
        elif filename.endswith('.zarr'):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, k) for k in range(num_frames)]
                for _ in tqdm(
                    as_completed(futures),
                    total=num_frames,
                    desc='Processing images',
                ):
                    _.result()

        success = True
    except Exception as e:
        logger.error(f'Failed to save registered stack: {e}')
    finally:
        if filename.endswith('.tif'):
            writer.close()
            if isinstance(moving_stack, TiffSeqHandler):
                with tf.TiffFile(moving_stack._tiff_seq.files[0]) as fl:
                    if fl.is_ome:
                        obj = ome.OME.from_xml(fl.ome_metadata)
                        obj.images[0].name = 'Registered Stack'
                        obj.images[0].description = 'Registered image stack'
                        obj.images[0].pixels.size_c = channels
                        tf.tiffcomment(filename, obj.to_xml())
                    else:
                        tf.tiffcomment(
                            filename,
                            moving_stack.minimal_metadata(
                                name='Registered Stack',
                                description='Registered image stack',
                                channels=channels,
                            ),
                        )
            else:
                tf.tiffcomment(
                    filename,
                    moving_stack.minimal_metadata(
                        name='Registered Stack',
                        description='Registered image stack',
                        channels=channels,
                    ),
                )
        elif filename.endswith('.zarr'):
            writer.store.close()

    return success


class RegistrationWidget(QtWidgets.QDialog):
    NAME = 'Registration Tool'

    def __init__(self, parent=None, stacks: dict[str, ref[ImageSequenceBase]] = None):
        super().__init__(parent)

        self.setMinimumWidth(500)

        self.stacks: dict[str, ref[ImageSequenceBase]] = (
            stacks if isinstance(stacks, dict) else {}
        )

        self._stack_reg = None
        self._transformation = None
        self._overlaid_img = None

        self.setWindowTitle('Image Registration')

        self.reference_stack = QtWidgets.QComboBox()
        self.moving_stack = QtWidgets.QComboBox()
        self.reference_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.moving_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.reference_stack.addItems(self.stacks.keys())
        self.moving_stack.addItems(self.stacks.keys())
        self.reference_stack.setCurrentIndex(0)
        self.moving_stack.setCurrentIndex(1 if len(self.stacks) > 1 else 0)

        self.method_label = QtWidgets.QLabel('Method:')
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(
            [method.name.replace('_', ' ').title() for method in RegistrationMethod]
        )
        self.method_combo.setCurrentText(
            RegistrationMethod.AFFINE.name.replace('_', ' ').title()
        )

        self.register_button = QtWidgets.QPushButton('Register')
        self.register_button.clicked.connect(self.register_images)

        self.transform_button = QtWidgets.QPushButton('Apply Transformation')
        self.transform_button.clicked.connect(self.apply_transformation)

        self.reset_button = QtWidgets.QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_transformation)

        self.autolevels_checkbox = QtWidgets.QCheckBox('Auto Levels')
        self.autolevels_checkbox.setChecked(True)

        self.histogram_checkbox = QtWidgets.QCheckBox('Show Histogram')
        self.histogram_checkbox.setChecked(True)

        self.stats_checkbox = QtWidgets.QCheckBox('Show Stats')
        self.stats_checkbox.setChecked(False)

        self.open_cvt_checkbox = QtWidgets.QCheckBox('Use OpenCV for Transformation')
        self.open_cvt_checkbox.setChecked(False)

        self.import_button = QtWidgets.QPushButton('Import Transformation')
        self.import_button.clicked.connect(self.import_transformation)

        self.export_button = QtWidgets.QPushButton('Export Transformation')
        self.export_button.clicked.connect(self.export_transformation)

        self.register_stack_button = QtWidgets.QPushButton('Register Stack')
        self.register_stack_button.clicked.connect(self.register_stack)

        layout = QtWidgets.QFormLayout()
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        layout.addRow('Reference Stack:', self.reference_stack)
        layout.addRow('Moving Stack:', self.moving_stack)
        layout.addRow(self.method_label, self.method_combo)
        layout.addRow('', self.autolevels_checkbox)
        layout.addRow('', self.histogram_checkbox)
        layout.addRow('', self.stats_checkbox)
        layout.addRow('', self.open_cvt_checkbox)
        layout.addRow(self.register_button)
        layout.addRow(self.transform_button)
        layout.addRow(self.reset_button)
        layout.addRow(self.import_button)
        layout.addRow(self.export_button)
        layout.addRow(self.register_stack_button)

        self.setLayout(layout)

        self.reference_stack.currentIndexChanged.connect(self.update_overlay)
        self.moving_stack.currentIndexChanged.connect(self.update_overlay)

        self.update_overlay()

        # timer to refresh display
        self._timer_interval = 100  # 10 FPS
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update_display)
        self._timer.setSingleShot(True)
        self._timer.start(self._timer_interval)

    def _stack(self, reference: bool = True) -> ImageSequenceBase | None:
        stack_name = (
            self.reference_stack.currentText()
            if reference
            else self.moving_stack.currentText()
        )
        stack_ref = self.stacks.get(stack_name, None)
        return stack_ref() if stack_ref is not None else None

    def _channel(self, reference: bool = True) -> int:
        stack_name = (
            self.reference_stack.currentText()
            if reference
            else self.moving_stack.currentText()
        )
        match = re.search(r'\(C(\d+)\)', stack_name)
        return int(match.group(1)) if match else 0

    def _get_slice(self, idx: int, reference: bool = True) -> np.ndarray | None:
        stack = self._stack(reference)
        if stack is not None:
            return stack.getSlice(idx, self._channel(reference))
        return None

    def update_display(self):
        if self._overlaid_img is not None and isinstance(
            self._overlaid_img, np.ndarray
        ):
            display_frame(
                RegistrationWidget.NAME,
                self._overlaid_img,
                histogram=self.histogram_checkbox.isChecked(),
                autoLevels=self.autolevels_checkbox.isChecked(),
                stats=self.stats_checkbox.isChecked(),
            )

        enabled = self._transformation is not None
        self.transform_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)

        self._timer.start(self._timer_interval)

    def update_overlay(self, img2: np.ndarray = None):
        img1: np.ndarray = self._get_slice(0).copy()
        if not isinstance(img2, np.ndarray):
            img2 = self._get_slice(0, reference=False).copy()

        if img1.shape != img2.shape:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                'Images must have the same dimensions for overlay.',
            )
            return

        self._overlaid_img = np.zeros(img1.shape[:2] + (3,), dtype=img1.dtype)
        self._overlaid_img[..., 0] = img1
        self._overlaid_img[..., 2] = img2

    def apply_transformation(self):
        if self._transformation is None or self._stack_reg is None:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                'No transformation available. Please register images first.',
            )
            return

        img = self._get_slice(0, reference=False).squeeze().copy()

        if self.open_cvt_checkbox.isChecked():
            # transform using cv2 and perserve range
            transformed_img = cv2_warp(img, self._transformation)
        else:
            transformed_img = self._stack_reg.transform(img)

        self.update_overlay(transformed_img)

    def reset_transformation(self):
        self._transformation = None
        self._stack_reg = None
        self.update_overlay()

    def register_images(self):
        method = RegistrationMethod.from_string(self.method_combo.currentText())
        logger.info(f'Registering images using method: {method.name}')

        if self.reference_stack.currentText() == self.moving_stack.currentText():
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Reference and moving stacks must be different.'
            )
            return

        img1: np.ndarray = np.nanmean(self._get_slice(slice(0, 10)).squeeze(), axis=0)
        img2: np.ndarray = np.nanmean(
            self._get_slice(slice(0, 10), reference=False).squeeze(), axis=0
        )

        if img1.shape != img2.shape:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                'Images must have the same dimensions for overlay.',
            )
            return

        self._stack_reg = StackReg(method.value)
        self._transformation = self._stack_reg.register(img1, img2)

        logger.info('Registration complete.')
        # Here you would typically update the displayed images in the GUI

    def register_stack(self):
        if self._transformation is None or self._stack_reg is None:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                'No transformation available. Please register images first.',
            )
            return

        # directory of moving stack
        path = self._stack(reference=False).path
        directory = os.path.dirname(path)

        # get save zarr file path
        file_name, _ = getSaveFileName(
            self,
            'Save Registered Stack',
            directory,
            'Zarr Files (*.zarr);;TIFF Files (*.tif)',
        )

        if not file_name:
            return

        ref_stack = self._stack()
        moving_stack = self._stack(reference=False)

        merge = False
        # bool if ref and moving shapes are equal
        if all(
            ref_stack.shapeTCZYX()[i] == moving_stack.shapeTCZYX()[i]
            for i in [0, 2, 3, 4]
        ):
            merge = (
                QtWidgets.QMessageBox.question(
                    self,
                    'Merge Stacks',
                    'The reference and moving stacks have the same shape.\n'
                    'Do you want to merge them into a single multi-channel stack?',
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                == QtWidgets.QMessageBox.StandardButton.Yes
            )

        self.register_stack_button.setEnabled(False)
        self.register_stack_button.setText('Registering...')

        def done(result):
            self.register_stack_button.setEnabled(True)
            self.register_stack_button.setText('Register Stack')
            if result:
                QtWidgets.QMessageBox.information(
                    self,
                    'Success',
                    f'Registered stack saved to:\n{file_name}',
                )

        worker = QThreadWorker(
            process_stack,
            file_name,
            ref_stack if merge else None,
            moving_stack,
            self._transformation,
            self._channel(True),
            self._channel(False),
        )
        worker.signals.result.connect(done)

        QtCore.QThreadPool.globalInstance().start(worker)

    def import_transformation(self):
        file_name, _ = getOpenFileName(
            self,
            'Import Transformation Matrix',
            '',
            'NumPy Files (*.npz)',
        )
        if file_name:
            try:
                data = np.load(file_name)
                self._transformation = data['matrix']
                method = data['method']
                method_enum = RegistrationMethod(method)
                if method_enum is None:
                    raise ValueError('Invalid registration method selected.')
                self._stack_reg = StackReg(method_enum.value)
                self._stack_reg.set_matrix(self._transformation)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    'Error',
                    f'Failed to import transformation: {e}',
                )

    def export_transformation(self):
        if self._transformation is None:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'No transformation to export.'
            )
            return

        # Save as .npz file
        file_name, _ = getSaveFileName(
            self,
            'Export Transformation Matrix',
            '',
            'NumPy Files (*.npz)',
        )
        if file_name:
            np.savez(
                file_name,
                matrix=self._stack_reg.get_matrix(),
                method=self._stack_reg._transformation,
            )
            QtWidgets.QMessageBox.information(
                self, 'Success', 'Transformation matrix exported successfully.'
            )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        DisplayManager.instance().close_display(RegistrationWidget.NAME)
        return super().closeEvent(event)


if __name__ == '__main__':
    import sys

    from microEye.qt import QApplication

    app = QApplication(sys.argv)

    # Example usage with dummy data
    stack1 = TiffSeqHandler(
        tf.TiffSequence(
            'F:/Nour (Do Not Delete)/'
            '2025_07_16/17_488_638_7f1f640d_200ms_tsBeads_testing3_minimalDrift/'
            '00__image_00000_roi_00.ome.tif'
        )
    )
    stack1.open()
    stack2 = TiffSeqHandler(
        tf.TiffSequence(
            'F:/Nour (Do Not Delete)/'
            '2025_07_16/17_488_638_7f1f640d_200ms_tsBeads_testing3_minimalDrift/'
            '00__image_00000_roi_01.ome.tif'
        )
    )
    stack2.open()

    stack3 = TiffSeqHandler(
        tf.TiffSequence(
            'F:/Nour (Do Not Delete)/'
            '2025_07_16/17_488_638_7f1f640d_200ms_tsBeads_testing3_minimalDrift/'
            'registered.tif'
        )
    )
    stack3.open()

    stacks = {
        'Stack 1': ref(stack1),
        'Stack 2': ref(stack2),
        'Stack 3 (C0)': ref(stack3),
        'Stack 3 (C1)': ref(stack3),
    }

    reg_widget = RegistrationWidget(stacks=stacks)
    reg_widget.show()

    sys.exit(app.exec())
