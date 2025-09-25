from microEye.hardware.stages.manager import Axis, StageManager, Units
from microEye.qt import QApplication, Qt, QtCore, QtGui, QtWidgets, Signal

SLIDES = {
    'sticky-Slide VI 0.4': {
        'width': 755,  # pixels (1 pixel = 0.1 mm)
        'height': 255,  # pixels (1 pixel = 0.1 mm)
        'channels': [
            # Each channel is defined by (x, y, width, height)
            # x, y are the top-left corner of the channel rectangle
            # rounded: (rx, ry) for rounded corners
            # If rounded is not provided, corners are not rounded
            # Coordinates are in pixels (1 pixel = 0.1 mm)
            {'id': 1, 'rect': (152 - 19, 37, 38, 180.5), 'rounded': (38 / 2, 38 / 2)},
            {
                'id': 2,
                'rect': (152 + 90 - 19, 37, 38, 180.5),
                'rounded': (38 / 2, 38 / 2),
            },
            {
                'id': 3,
                'rect': (152 + 180 - 19, 37, 38, 180.5),
                'rounded': (38 / 2, 38 / 2),
            },
            {
                'id': 4,
                'rect': (152 + 270 - 19, 37, 38, 180.5),
                'rounded': (38 / 2, 38 / 2),
            },
            {
                'id': 5,
                'rect': (152 + 360 - 19, 37, 38, 180.5),
                'rounded': (38 / 2, 38 / 2),
            },
            {
                'id': 6,
                'rect': (152 + 450 - 19, 37, 38, 180.5),
                'rounded': (38 / 2, 38 / 2),
            },
        ],
    },
    'μ-Slide VI 0.1': {
        'width': 755,  # pixels (1 pixel = 0.1 mm
        'height': 255,  # pixels (1 pixel = 0.1 mm)
        'channels': [
            # Each channel is defined by (x, y, width, height)
            # x, y are the top-left corner of the channel rectangle
            # rounded: (rx, ry) for rounded corners
            # If rounded is not provided, corners are not rounded
            # Coordinates are in pixels (1 pixel = 0.1 mm)
            {'id': 1, 'rect': (152 - 5, 37, 10, 180.5), 'rounded': (5, 5)},
            {
                'id': 2,
                'rect': (152 + 90 - 5, 37, 10, 180.5),
                'rounded': (5, 5),
            },
            {
                'id': 3,
                'rect': (152 + 180 - 5, 37, 10, 180.5),
                'rounded': (5, 5),
            },
            {
                'id': 4,
                'rect': (152 + 270 - 5, 37, 10, 180.5),
                'rounded': (5, 5),
            },
            {
                'id': 5,
                'rect': (152 + 360 - 5, 37, 10, 180.5),
                'rounded': (5, 5),
            },
            {
                'id': 6,
                'rect': (152 + 450 - 5, 37, 10, 180.5),
                'rounded': (5, 5),
            },
        ],
    },
    'sticky-Slide 8 Well': {
        'width': 755,  # pixels (1 pixel = 0.1 mm)
        'height': 255,  # pixels (1 pixel = 0.1 mm)
        'channels': [],
    },
    'sticky-Slide 18 Well': {
        'width': 755,  # pixels (1 pixel = 0.1 mm)
        'height': 255,  # pixels (1 pixel = 0.1 mm)
        'channels': [],
    },
}
SLIDES['μ-Slide VI 0.4'] = SLIDES['sticky-Slide VI 0.4'].copy()

for y in (71.5, 183.5):
    for x in (190 + i * 125 for i in range(4)):
        SLIDES['sticky-Slide 8 Well']['channels'].append(
            {
                'id': len(SLIDES['sticky-Slide 8 Well']['channels']) + 1,
                'rect': (x - 106.5 / 2, y - 94.1 / 2, 106.5, 94.1),
                'rounded': False,
            }
        )

for y in (53, 127.5, 202):
    for x in (175 + i * 81 for i in range(6)):
        SLIDES['sticky-Slide 18 Well']['channels'].append(
            {
                'id': len(SLIDES['sticky-Slide 18 Well']['channels']) + 1,
                'rect': (x - 61 / 2, y - 57 / 2, 61, 57),
                'rounded': False,
            }
        )


class ChannelSignals(QtCore.QObject):
    selected = Signal(int)  # channel_id


class ChannelItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, path, channel_id):
        super().__init__(path)
        self.channel_id = channel_id
        self.signals = ChannelSignals()
        self.setBrush(QtGui.QBrush(QtGui.QColor('lightblue')))
        self.setPen(QtGui.QPen(QtGui.QColor('#444444')))
        # self.setFlag(self.GraphicsItemFlag.Itemis, True)
        self.is_selected = False

        # draw id
        font = QtGui.QFont()
        font.setPointSize(10)
        self.text_item = QtWidgets.QGraphicsTextItem(str(channel_id), self)
        self.text_item.setDefaultTextColor(QtGui.QColor('black'))
        self.text_item.setZValue(1)

    def set_selected(self, selected):
        self.is_selected = selected
        color = 'orange' if selected else 'lightblue'
        pen = '#A35A00' if selected else '#444444'
        self.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        self.setPen(QtGui.QPen(QtGui.QColor(pen)))

    def mousePressEvent(self, event):
        self.signals.selected.emit(self.channel_id)
        super().mousePressEvent(event)

    def contains_point(self, point: QtCore.QPointF):
        return self.path().contains(point)

    @property
    def center(self):
        '''Return the center point of the channel in pixels relative to the slide.'''
        return self.path().boundingRect().center()


class SlideWidget(QtWidgets.QGraphicsView):
    channel_selected = Signal(int, float, float)  # channel_id, center_x, center_y

    INFO_HEIGHT = 30  # pixels

    def __init__(self, slide: str = None, parent=None):
        """
        A widget to visualize a slide with channels and a movable position cross.

        **Notes:**
        - that the coordinate system has (0,0) at the top-left corner,
          with x increasing to the right and y increasing downwards.
        - The position cross can be updated using the `set_current_position` method.
        - Dimensions and positions are in pixels and converted to mm based on a scale
          factor of 1 pixel = 0.1 mm.

        Parameters
        ----------
        slide : str
            The type of slide to visualize. Supported types are:
            - 'sticky-Slide VI 0.4'
            - 'μ-Slide VI 0.4'
            - 'μ-Slide VI 0.1'
        """
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene()
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        if slide not in SLIDES:
            slide = 'sticky-Slide VI 0.4'  # Default slide

        self._slide = slide
        self.channels: dict[int, ChannelItem] = {}
        self.current_pos = QtCore.QPointF(self.slide_width / 2, self.slide_height / 2)
        self.selected_channel_id = None

        self._swap_xy = False
        self._invert_x = False
        self._invert_y = False

        self._draw_slide()
        self._draw_info_bar()
        self._draw_channels()

        # context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)
        slide_menu = menu.addMenu('Select Slide')
        slide_menu.setToolTip('Select the type of slide to visualize.')
        for key in sorted(SLIDES):
            action = slide_menu.addAction(key)
            action.setCheckable(True)
            action.setChecked(key == self._slide)
            action.triggered.connect(lambda checked, k=key: self._set_slide(k))

        swap_action = menu.addAction('Swap X/Y')
        swap_action.setCheckable(True)
        swap_action.setChecked(self._swap_xy)
        swap_action.setToolTip(
            'Swap the X and Y cordinates to match the stage orientation.'
        )
        swap_action.triggered.connect(self._toggle_swap_xy)

        invert_x_action = menu.addAction('Invert X')
        invert_x_action.setCheckable(True)
        invert_x_action.setChecked(self._invert_x)
        invert_x_action.triggered.connect(self._toggle_invert_x)
        invert_x_action.setToolTip('Invert the X coordinate to match the stage.')

        invert_y_action = menu.addAction('Invert Y')
        invert_y_action.setCheckable(True)
        invert_y_action.setChecked(self._invert_y)
        invert_y_action.triggered.connect(self._toggle_invert_y)
        invert_y_action.setToolTip('Invert the Y coordinate to match the stage.')
        # Show the menu at the cursor position
        menu.exec(self.mapToGlobal(pos))

    def _toggle_swap_xy(self):
        self._swap_xy = not self._swap_xy

    def _toggle_invert_x(self):
        self._invert_x = not self._invert_x

    def _toggle_invert_y(self):
        self._invert_y = not self._invert_y

    def _set_slide(self, slide: str):
        if slide not in SLIDES:
            raise ValueError(f'Unsupported slide type: {slide}')
        self._slide = slide
        self._draw_channels()
        self.fitInView(
            0,
            0,
            self.slide_width,
            self.slide_height,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(
            0,
            0,
            self.slide_width,
            self.slide_height,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    @property
    def slide_width(self):
        return SLIDES[self._slide]['width']

    @property
    def slide_height(self):
        return SLIDES[self._slide]['height']

    @property
    def center(self):
        '''Return the center point of the slide in pixels.'''
        return QtCore.QPointF(self.slide_width / 2, self.slide_height / 2)

    def _draw_slide(self):
        slide_path = QtGui.QPainterPath()
        slide_path.addRect(0, 0, self.slide_width, self.slide_height)
        slide_item = QtWidgets.QGraphicsPathItem(slide_path)
        slide_item.setBrush(QtGui.QBrush(QtGui.QColor('darkgray')))
        self._scene.addItem(slide_item)

    def _draw_info_bar(self):
        # Draw a bar at the bottom with slide info
        info_path = QtGui.QPainterPath()
        info_path.addRect(
            0, self.slide_height, self.slide_width, SlideWidget.INFO_HEIGHT
        )
        info_item = QtWidgets.QGraphicsPathItem(info_path)
        info_item.setBrush(QtGui.QBrush(QtGui.QColor('#222222')))
        self._scene.addItem(info_item)
        # Add text
        font = QtGui.QFont()
        font.setPointSize(10)
        self._label = QtWidgets.QGraphicsTextItem('')
        self._label.setDefaultTextColor(QtGui.QColor('white'))
        self._label.setFont(font)
        self._scene.addItem(self._label)

        self._update_info_bar()

    def _update_info_bar(self):
        stage = StageManager.instance().xy_stage()

        if stage is None:
            stage_info = 'No XY Stage'
        else:
            stage_info = (
                f'X: {stage.x:.2f} {stage.get_unit(Axis.X).suffix()} '
                f'Y: {stage.y:.2f} {stage.get_unit(Axis.Y).suffix()} | '
                f'(dX: {stage.dx:.2f} {stage.get_unit(Axis.X).suffix()}, '
                f'dY: {stage.dy:.2f} {stage.get_unit(Axis.Y).suffix()})'
            )

        self._label.setPlainText(stage_info)

        self._label.setPos(
            (self.slide_width - self._label.boundingRect().width()) / 2,
            self.slide_height
            + (SlideWidget.INFO_HEIGHT - self._label.boundingRect().height()) / 2,
        )

    def _draw_center_cross(self):
        # Remove previous cross if exists
        if hasattr(self, 'center_cross_items'):
            for item in self.center_cross_items:
                self._scene.removeItem(item)
        pen = QtGui.QPen(QtGui.QColor('#444444'), 1)
        h_line = self._scene.addLine(
            self.slide_width / 2 - 20,
            self.slide_height / 2,
            self.slide_width / 2 + 20,
            self.slide_height / 2,
            pen,
        )
        v_line = self._scene.addLine(
            self.slide_width / 2,
            self.slide_height / 2 - 20,
            self.slide_width / 2,
            self.slide_height / 2 + 20,
            pen,
        )
        ellipse = self._scene.addEllipse(
            self.slide_width / 2 - 5, self.slide_height / 2 - 5, 10, 10, pen
        )
        self.center_cross_items = [h_line, v_line, ellipse]

    def _draw_position_cross(self):
        # Remove previous cross if exists
        if hasattr(self, 'pos_cross_items'):
            for item in self.pos_cross_items:
                self._scene.removeItem(item)
        pen = QtGui.QPen(QtGui.QColor('blue'), 1)
        x, y = self.current_pos.x(), self.current_pos.y()
        h_line = self._scene.addLine(x - 10, y, x + 10, y, pen)
        v_line = self._scene.addLine(x, y - 10, x, y + 10, pen)
        ellipse = self._scene.addEllipse(x - 3, y - 3, 6, 6, pen)
        self.pos_cross_items = [h_line, v_line, ellipse]

    def _draw_channels(self):
        self._remove_channels()

        for channel_def in SLIDES[self._slide]['channels']:
            rect_x, rect_y, rect_w, rect_h = channel_def['rect']

            rounded = channel_def.get('rounded', False)

            path = QtGui.QPainterPath()
            # Channel body
            if isinstance(rounded, (list, tuple)) and len(rounded) == 2:
                rx, ry = rounded
                path.addRoundedRect(
                    rect_x,
                    rect_y - rect_w / 2,
                    rect_w,
                    rect_h + rect_w,
                    rx,
                    ry,
                )  # Rounded corners
            else:
                path.addRect(rect_x, rect_y, rect_w, rect_h)

            channel = ChannelItem(path, channel_id=channel_def['id'])
            channel.text_item.setPos(
                rect_x + rect_w / 2 - channel.text_item.boundingRect().width() / 2,
                rect_y + rect_h / 2 - channel.text_item.boundingRect().height() / 2,
            )
            channel.signals.selected.connect(self._on_channel_selected)
            self._scene.addItem(channel)
            self.channels[channel.channel_id] = channel

        self._draw_center_cross()
        self._draw_position_cross()
        self._update_channel_highlight()

    def _remove_channels(self):
        for channel in self.channels.values():
            channel.signals.selected.disconnect(self._on_channel_selected)
            self._scene.removeItem(channel)
        self.channels.clear()
        self.selected_channel_id = None

    def set_current_position(self, x, y, pixels: bool = True):
        if not pixels:
            x /= 100  # Convert micrometers to pixels
            y /= 100
        if self._swap_xy:
            x, y = y, x

        if self._invert_x:
            x = self.slide_width - x
        if self._invert_y:
            y = self.slide_height - y

        x = max(0, min(self.slide_width, x))
        y = max(0, min(self.slide_height, y))

        self.current_pos = QtCore.QPointF(x, y)
        self._draw_position_cross()
        self._update_channel_highlight()
        self._update_info_bar()

    def _on_channel_selected(self, channel_id):
        self.selected_channel_id = channel_id
        channel = self.channels.get(channel_id)
        if channel is None:
            return

        x, y = channel.center.x(), channel.center.y()

        if self._swap_xy:
            x, y = y, x
        if self._invert_x:
            x = self.slide_width - x
        if self._invert_y:
            y = self.slide_height - y

        x -= self.center.x()
        y -= self.center.y()

        self.channel_selected.emit(channel_id, x, y)

        stage = StageManager.instance().xy_stage()

        if stage is None:
            return

        x = Units.convert(
            x,
            Units.SLIDES_PIXELS,
            stage.get_unit(Axis.X),
        ) + stage.get_center(Axis.X)

        y = Units.convert(
            y,
            Units.SLIDES_PIXELS,
            stage.get_unit(Axis.Y),
        ) + stage.get_center(Axis.Y)

        # yes no dialog to move stage to channel center
        reply = QtWidgets.QMessageBox.question(
            self,
            'Move Stage',
            f'Move stage to center of channel {channel_id}?'
            f'\nX: {x} {stage.get_unit(Axis.X).suffix()},'
            f' Y: {y} {stage.get_unit(Axis.Y).suffix()}',
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Move the stage to the center of the channel
            StageManager.instance().move_absolute(x, y)

        self._update_channel_highlight()

    def _update_channel_highlight(self):
        for _, channel in self.channels.items():
            # Highlight if selected or if current position is inside
            inside = channel.contains_point(self.current_pos)
            # selected = channel_id == self.selected_channel_id
            channel.set_selected(inside)

    def update_position(self):
        '''Update the position cross based on the current stage position.'''
        stage = StageManager.instance().xy_stage()
        if stage is None:
            return

        if Axis.X not in stage.axes or Axis.Y not in stage.axes:
            return

        x = (
            Units.convert(
                stage.dx,
                stage.get_unit(Axis.X),
                Units.SLIDES_PIXELS,
            )
            + self.center.x()
        )
        y = (
            Units.convert(
                stage.dy,
                stage.get_unit(Axis.Y),
                Units.SLIDES_PIXELS,
            )
            + self.center.y()
        )

        self.set_current_position(x, y, pixels=True)

    def get_config(self) -> dict:
        return {
            'slide': self._slide,
            'swap_xy': self._swap_xy,
            'invert_x': self._invert_x,
            'invert_y': self._invert_y,
        }

    def load_config(self, config: dict):
        slide = config.get('slide', self._slide)
        self._set_slide(slide)

        self._swap_xy = config.get('swap_xy', self._swap_xy)
        self._invert_x = config.get('invert_x', self._invert_x)
        self._invert_y = config.get('invert_y', self._invert_y)


# Example usage
if __name__ == '__main__':
    app = QApplication([])

    widget = SlideWidget()
    widget.setWindowTitle('μ-Slide VI 0.4 Geometry Example')
    widget.resize(800, 400)
    widget.show()

    # Example: update position cross after 2 seconds

    def move_cross():
        widget.set_current_position(152 + 90 * 2, 100)

    QtCore.QTimer.singleShot(5000, move_cross)

    app.exec()
