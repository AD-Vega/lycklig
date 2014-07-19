#!/usr/bin/env python

# kinky, a wavy image enhancer
# Copyright (C) 2014 Jure Varlec <jure.varlec@ad-vega.si>
#                    Andrej Lajovic <andrej.lajovic@ad-vega.si>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from PyQt4.QtGui import QApplication, QGraphicsView, QGraphicsScene, QPixmap, QImage, \
    QGridLayout, QVBoxLayout, QHBoxLayout, QStackedLayout, QLabel, QWidget, \
    QPalette, QFileDialog, QMessageBox, QTransform, QTextDocument, QMouseEvent
from PyQt4.QtCore import Qt, QEvent, QPoint, QTimer, QSize
from base64 import b64decode, b64encode
from scipy import ndimage
import numpy as np
import PythonMagick
import sys, os, math

_k_enh_prec = '{:.5g}'
_σ_enh_prec = '{:.2f}'
_σ_noise_prec = '{:.2f}'

_k_enh_default = 1.0
_σ_enh_default = 0.25
_σ_noise_default = 0.1

def numpy2QImage(arrayImage):
    height, width, depth = arrayImage.shape
    arrayImage = (arrayImage.astype('float') * 255 / arrayImage.max()).astype('uint8')
    if depth == 3:
        qimg = QImage(width, height, QImage.Format_RGB888)
    else:
        qimg = QImage(width, height, QImage.Format_Indexed8)
    rawsize = height * qimg.bytesPerLine()
    rawptr = qimg.bits()
    rawptr.setsize(rawsize)
    rawarr = np.frombuffer(memoryview(rawptr), dtype='uint8')
    arr = rawarr.reshape((height, qimg.bytesPerLine()))
    arr[:, 0:(width*depth)] = arrayImage[:, :, :].reshape((height, width*depth))
    return qimg

class OpaqueLabel(QLabel):
    def __init__(self, text = ""):
        super().__init__()
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)
        self.setText(text)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

class HelpWidget(QWidget):
    def __init__(self):
        super().__init__()
        txt = """
        <h1>Welcome to Kinky, the Wavy Image Enhancer!</h1>
        <p>
        <ul>
        <li>Click to dismiss this help text.</li>
        <li>Drag holding the middle mouse button to pan view.</li>
        <li>Click and drag to:
          <ul>
          <li>enhance the image by dragging up/down,</li>
          <li>denoise the image by dragging left/right,</li>
          <li>set the finesse of enhancement by dragging left/right
              while also holding the right mouse button.</li>
          </ul>
        <li>Hold Shift while dragging for greater precision.</li>
        <li>Zoom using mouse wheel or +/- keys.</li>
        <li>Press = or 0 to unzoom.</li>
        <li>Hold Tab to show the original image.</li>
        <li>Press R to reset the image back to original.</li>
        <li>Press H to show this text again.</li>
        <li>Press Q or Esc to quit (Q will ask to save the image).</li>
        <li>Press S to save the image.</li>
        <li>Run this program with the --help argument to learn about batch mode.</li>
        </ul>
        </p>
        """
        label = OpaqueLabel(txt)
        tdoc = QTextDocument()
        tdoc.setHtml('<h1>the Wavy Image Enhancer!</h1>')
        width = tdoc.idealWidth()
        tdoc.setHtml(txt)
        tdoc.setTextWidth(width)
        size = tdoc.size()
        margin = 20
        size = QSize(size.width() + 2*margin, size.height() + 2*margin)
        label.setMargin(margin)
        label.setMaximumSize(size)
        label.setMinimumSize(size)
        label.setWordWrap(True)
        self.setLayout(QGridLayout())
        self.layout().setAlignment(Qt.AlignCenter)
        self.layout().addWidget(label, 1, 1)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setMinimumSize(label.sizeHint())

class LabelsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._dlabel = OpaqueLabel()
        self._klabel = OpaqueLabel()
        self._elabel = OpaqueLabel()
        self._nlabel = OpaqueLabel()
        self.updateLabels(0.0, 0.0, 0.0)
        self.setLayout(QHBoxLayout())
        self.layout().setAlignment(Qt.AlignTop | Qt.AlignLeft)
        col = QVBoxLayout()
        col.addWidget(self._klabel)
        col.addWidget(self._elabel)
        col.addWidget(self._nlabel)
        col.addWidget(self._dlabel)
        self.layout().addLayout(col)
        self._dlabel.setVisible(False)

    def showDepth(self, depth):
        self._dlabel.setText(self._depthText.format(depth))
        self._dlabel.setVisible(True)

    def updateLabels(self, k_enh, σ_enh, σ_noise):
        self._klabel.setText(self._klabelText.format(k_enh))
        self._elabel.setText(self._elabelText.format(σ_enh))
        self._nlabel.setText(self._nlabelText.format(σ_noise))

    _depthText = "Image depth: {}-bit"
    _klabelText = "Enhancement k: " + _k_enh_prec
    _elabelText = "Enhancement σ: " + _σ_enh_prec
    _nlabelText = "Denoising σ: " + _σ_noise_prec

class ImageEnhancer(QGraphicsView):
    def __init__(self, imagefile):
        super().__init__()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setWindowTitle("Kinky: " + imagefile)
        self._timer.timeout.connect(self.updateImage)
        self._timer.setSingleShot(True)

        # Prepare the overlay display widgets
        mainlayout = QStackedLayout()
        mainlayout.setStackingMode(QStackedLayout.StackAll)
        self.setLayout(mainlayout)
        self._help = HelpWidget()
        self._overlay = LabelsWidget()
        mainlayout.addWidget(self._overlay)
        mainlayout.addWidget(self._help)
        mainlayout.setCurrentWidget(self._help)

        # Load and display image
        self._filename = imagefile
        image = PythonMagick.Image(imagefile)
        self._overlay.showDepth(image.depth())
        blob = PythonMagick.Blob()
        # ImageMagick always makes color images ...
        image.write(blob, 'RGB', 16)
        rawdata = b64decode(blob.base64())
        self._img = np.ndarray((image.rows(), image.columns(), 3),
                               dtype='uint16', buffer=rawdata)
        # ... so make them grayscale if necessary
        if (self._img[:,:,0] == self._img[:,:,1]).all() \
            and (self._img[:,:,2] == self._img[:,:,1]).all():
            tmp = np.ndarray((image.rows(), image.columns(), 1), dtype='uint16')
            tmp[:,:,0] = self._img[:,:,0]
            self._img = tmp
        self._img = self._img.astype('float')
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(Qt.black)
        self._qimg = numpy2QImage(self._img)
        self._pic = self._scene.addPixmap(QPixmap.fromImage(self._qimg))
        self.setScene(self._scene)
        self._newimg = self._img

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._doNotOperate:
            self.setDragMode(QGraphicsView.NoDrag)
            self._lastPos = event.pos()
            self._help.setVisible(False)
            event.accept()
        elif event.button() == Qt.MiddleButton:
            event = QMouseEvent(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._lastPos = QPoint()
            event.accept()
        elif event.button() == Qt.MiddleButton:
            event = QMouseEvent(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and not self._doNotOperate:
            self._saved = False
            dp = event.pos() - self._lastPos
            self._lastPos = event.pos()
            self._k_enh *= 10**(-dp.y() / self._exp_factor)
            if event.buttons() & Qt.RightButton:
                self._σ_enh *= 10**(dp.x() / self._exp_factor)
            else:
                self._σ_noise *= 10**(dp.x() / self._exp_factor)
            self._overlay.updateLabels(self._k_enh, self._σ_enh, self._σ_noise)
            if not self._timer.isActive():
                self._timer.start(100) # delayed update of the image
            else:
                self._doUpdate = True
            event.accept()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        event.accept()
        self.zoom(event.delta() > 0)
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Q:
            buttons = QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            if self._saved:
                button = QMessageBox.Discard
            else:
                button = QMessageBox.question(self, "Save image?",
                                              "Do you want to save the image before exiting?",
                                              buttons, QMessageBox.Save)
            if button != QMessageBox.Cancel and button != QMessageBox.NoButton:
                success = True
                if button == QMessageBox.Save:
                    success = self.saveImage()
                if success:
                    self.close()
        elif event.key() == Qt.Key_S:
            self.saveImage()
        elif event.key() == Qt.Key_H:
            self._help.setVisible(True)
        elif event.key() == Qt.Key_Plus:
            self.zoom(True)
        elif event.key() == Qt.Key_Minus:
            self.zoom(False)
        elif event.key() == Qt.Key_0 or event.key() == Qt.Key_Equal:
            self.zoom(None)
        elif event.key() == Qt.Key_Shift:
            self._exp_factor *= 5.
        elif event.key() == Qt.Key_Tab:
            self._doNotOperate = True
            self._pic.setPixmap(QPixmap.fromImage(self._qimg))
        elif event.key() == Qt.Key_R:
            self._k_enh = _k_enh_default
            self._σ_enh = _σ_enh_default
            self._σ_noise = _σ_noise_default
            self._overlay.updateLabels(0.0, 0.0, 0.0)
            self._saved = True
            self.updateImage()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self._exp_factor /= 5.
        elif event.key() == Qt.Key_Tab:
            self._doNotOperate = False
            self._pic.setPixmap(QPixmap.fromImage(self._qnewimg))
        else:
            super().keyReleaseEvent(event)

    def saveImage(self):
        height, width, depth = self._newimg.shape
        if depth == 3:
            cmap = 'RGB'
        else:
            cmap = 'K'
        candidate, ext = os.path.splitext(self._filename)
        fmtstring = "_kay-{}_sigma-{}_noise-{}.png"
        candidate = candidate + fmtstring.format(_k_enh_prec, _σ_enh_prec, _σ_noise_prec)
        candidate = candidate.format(self._k_enh, self._σ_enh, self._σ_noise)
        namefilter = 'Image files (*.png *.tiff)'
        filename = QFileDialog.getSaveFileName(self, "Save Image", candidate, namefilter)
        if len(filename) > 0:
            blob = PythonMagick.Blob()
            u16arr = (self._newimg / self._newimg.max() * (2**16-1)).astype('uint16')
            blob.base64(b64encode(bytes(u16arr)).decode('ascii'))
            image = PythonMagick.Image(blob, PythonMagick.Geometry(width, height), 16, cmap)
            if depth == 1:
                pass
                # TODO: use image.type( GrayscaleType ), but the enum is not
                # exported in PythonMagick. Below commented code does nothing of value.
                #image.quantizeColorSpace(PythonMagick.ColorspaceType.GRAYColorspace)
                #image.quantizeColors(2**16)
                #image.quantize()
            error = ''
            try:
                image.write(filename)
            except Exception as e:
                error = 'Unexpected error: ' + str(e)
            if len(error) > 0:
                QMessageBox.warning(self, 'Error saving image', error)
                return False
            self._saved = True
            return True

    def updateImage(self):
        depth = self._img.shape[2]
        layer = np.empty_like(self._img)
        for ch in range(0, depth):
            layer[:,:,ch] = ndimage.gaussian_filter(self._img[:,:,ch], sigma=self._σ_noise)
            layer[:,:,ch] -= ndimage.gaussian_filter(layer[:,:,ch], sigma=self._σ_enh)

        self._newimg = np.fmax(0.0, self._img + self._k_enh * layer)
        self._qnewimg = numpy2QImage(self._newimg)
        self._pic.setPixmap(QPixmap.fromImage(self._qnewimg))
        if self._doUpdate:
            self._doUpdate = False
            self._timer.start()

    def zoom(self, what):
        if what == None:
            self.setTransform(QTransform())
        elif what == True:
            self.scale(1.1, 1.1)
        elif what == False:
            self.scale(0.9, 0.9)

    _k_enh = _k_enh_default
    _σ_enh = _σ_enh_default
    _σ_noise = _σ_noise_default
    _exp_factor = 200.
    _lastPos = QPoint()
    _timer = QTimer()
    _doUpdate = False
    _saved = True
    _doNotOperate = False

app = QApplication(sys.argv)
enh = ImageEnhancer(sys.argv[1])
enh.show()
retval = app.exec_()
del enh
del app
sys.exit(retval)
