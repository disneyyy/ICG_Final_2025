import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from vedo import Mesh, Plotter, settings
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay


class SketchCanvas(QtWidgets.QLabel):
    draw_finished = QtCore.pyqtSignal(list)  # emit when user presses enter

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(512, 512)
        self.setStyleSheet("background-color: black;")
        self.image = np.zeros((512, 512, 3), np.uint8)
        self.setPixmap(QtGui.QPixmap.fromImage(self.get_qimage()))
        self.drawing = False
        self.points = []
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()

    def get_qimage(self):
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

    def mousePressEvent(self, event):
        self.drawing = True
        self.points = [(event.x(), event.y())]

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.points.append((event.x(), event.y()))
            cv2.line(self.image, self.points[-2], self.points[-1], (255, 255, 255), 2)
            self.setPixmap(QtGui.QPixmap.fromImage(self.get_qimage()))

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return and len(self.points) > 10:
            self.draw_finished.emit(self.points)
            self.image[:] = 0
            self.points = []
            self.setPixmap(QtGui.QPixmap.fromImage(self.get_qimage()))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Sketch to 3D Generator")
        layout = QtWidgets.QHBoxLayout(self)

        # Left: sketch canvas
        self.canvas = SketchCanvas()
        layout.addWidget(self.canvas)

        # Right: 3D display using Plotter in external window (fallback)
        self.plotter = Plotter(title="3D View", axes=1, interactive=False)
        self.plotter.show(interactive=False)

        self.canvas.draw_finished.connect(self.add_model)
        self.models = []
        self.model_count = 0
        self.z_offset = 0

    def add_model(self, raw_points):
        mesh = self.contour_to_3d(raw_points, z_shift=self.z_offset)
        self.models.append(mesh)
        mesh.write(f"model_{self.model_count}.obj")
        self.model_count += 1
        self.z_offset -= 40
        self.plotter.clear()
        self.plotter.show(self.models, resetcam=False, viewup="z", interactive=False)

    def contour_to_3d(self, contour, z_shift=0):
        contour = np.array(contour)
        if np.linalg.norm(contour[0] - contour[-1]) > 5:
            contour = np.vstack([contour, contour[0]])

        k = min(3, len(contour) - 1)
        tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=True, k=k)
        unew = np.linspace(0, 1.0, 100)
        out = splev(unew, tck)
        resampled = np.vstack(out).T

        tri = Delaunay(resampled)
        center = np.mean(resampled, axis=0)
        max_dist = np.max(np.linalg.norm(resampled - center, axis=1))
        z_top = [(1 - (np.linalg.norm(p - center) / max_dist) ** 2) * 20 for p in resampled]
        z_top = np.array(z_top)
        z_bot = -z_top

        top_pts = np.hstack([resampled, z_top[:, np.newaxis]])
        bot_pts = np.hstack([resampled, z_bot[:, np.newaxis]])
        top_pts[:, 2] += z_shift
        bot_pts[:, 2] += z_shift

        pts = np.vstack([top_pts, bot_pts])
        faces = []

        for tri_pts in tri.simplices:
            faces.append([tri_pts[0], tri_pts[1], tri_pts[2]])
        for tri_pts in tri.simplices:
            a, b, c = tri_pts + len(resampled)
            faces.append([c, b, a])

        n = len(resampled)
        for i in range(n):
            a, b = i, (i + 1) % n
            a_bot, b_bot = a + n, b + n
            faces.append([a, b, b_bot])
            faces.append([a, b_bot, a_bot])

        mesh = Mesh([pts, faces])
        mesh.color("tomato").lighting("plastic").compute_normals()
        return mesh


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
