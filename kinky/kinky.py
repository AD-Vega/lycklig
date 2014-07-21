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
from PyQt4.QtCore import Qt, QEvent, QPoint, QTimer, QSize, QThread, \
    QDataStream, QByteArray, QIODevice
from base64 import b64decode, b64encode
from scipy import ndimage
from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Process, Pipe, Pool
import numpy as np
import PythonMagick
import sys, os, math

_k_enh_prec = '{:.5g}'
_σ_enh_prec = '{:.2f}'
_σ_noise_prec = '{:.2f}'
_filename_fmtstring = "_k{}_sigma{}_noise{}"
_filename_fmtstring = _filename_fmtstring.format(_k_enh_prec, _σ_enh_prec,
                                                 _σ_noise_prec)

_k_enh_default = 1.0
_σ_enh_default = 0.25
_σ_noise_default = 0.1

_zoom = 1.1

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

def loadImage(filename):
    image = PythonMagick.Image(filename)
    blob = PythonMagick.Blob()
    # ImageMagick always makes color images ...
    image.write(blob, 'RGB', 16)
    rawdata = b64decode(blob.base64())
    img = np.ndarray((image.rows(), image.columns(), 3),
                           dtype='uint16', buffer=rawdata)
    # ... so make them grayscale if necessary
    if (img[:,:,0] == img[:,:,1]).all() \
       and (img[:,:,2] == img[:,:,1]).all():
        tmp = np.ndarray((image.rows(), image.columns(), 1), dtype='uint16')
        tmp[:,:,0] = img[:,:,0]
        img = tmp
    img = img.astype('float')
    return img, image.depth()

def saveImage(image, filename):
    height, width, depth = image.shape
    if depth == 3:
        cmap = 'RGB'
    else:
        cmap = 'K'
    blob = PythonMagick.Blob()
    u16arr = (image / image.max() * (2**16-1)).astype('uint16')
    blob.base64(b64encode(bytes(u16arr)).decode('ascii'))
    image = PythonMagick.Image(blob, PythonMagick.Geometry(width, height), 16, cmap)
    if depth == 1:
        pass
        # TODO: use image.type( GrayscaleType ), but the enum is not
        # exported in PythonMagick. Below commented code does nothing of value.
        #image.quantizeColorSpace(PythonMagick.ColorspaceType.GRAYColorspace)
        #image.quantizeColors(2**16)
        #image.quantize()
    image.write(filename)

def processImage(img, k_enh, σ_enh, σ_noise):
    depth = img.shape[2]
    layer = np.empty_like(img)
    for ch in range(0, depth):
        layer[:,:,ch] = ndimage.gaussian_filter(img[:,:,ch], sigma=σ_noise)
        layer[:,:,ch] -= ndimage.gaussian_filter(layer[:,:,ch], sigma=σ_enh)
    return np.fmax(0.0, img + k_enh * layer)

def processImageAsync(*args):
    img = processImage(*args)
    qimg = numpy2QImage(img)
    pix = QPixmap.fromImage(qimg)
    b = QByteArray()
    s = QDataStream(b, QIODevice.WriteOnly)
    s.__lshift__(pix)
    return img, b

class AsyncFunction(QThread):
    def __init__(self, func, staticData = None):
        super().__init__()
        self._static = staticData
        self._in, self._out = Pipe()
        self._process = Process(target=AsyncFunction._processRun,
                                args=(func, self._static, self._out))
        self._process.start()

    def __del__(self):
        self._process.terminate()

    def apply(self, *args):
        self._args = args
        self.start()

    def result(self):
        return self._result

    def _processRun(func, staticData, pipe):
        while True:
            args = pipe.recv()
            if staticData == None:
                result = func(*args)
            else:
                result = func(staticData, *args)
            pipe.send(result)

    def run(self):
        self._in.send(self._args)
        self._result = self._in.recv()

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
        self._busyLabel = OpaqueLabel("Working...")
        self.updateLabels(0.0, 0.0, 0.0)
        self.setLayout(QHBoxLayout())
        self.layout().setAlignment(Qt.AlignTop | Qt.AlignLeft)
        col = QVBoxLayout()
        col.addWidget(self._klabel)
        col.addWidget(self._elabel)
        col.addWidget(self._nlabel)
        col.addWidget(self._dlabel)
        col.addWidget(self._busyLabel)
        self.layout().addLayout(col)
        self._dlabel.setVisible(False)
        self._busyLabel.setVisible(False)

    def showDepth(self, depth):
        self._dlabel.setText(self._depthText.format(depth))
        self._dlabel.setVisible(True)

    def showBusy(self, busy):
        self._busyLabel.setVisible(busy)

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
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setWindowTitle("Kinky: " + imagefile)

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
        self._img, depth = loadImage(imagefile)
        self._overlay.showDepth(depth)
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(Qt.black)
        qimg = numpy2QImage(self._img)
        self._pixmap = QPixmap.fromImage(qimg)
        self._pic = self._scene.addPixmap(self._pixmap)
        self.setScene(self._scene)
        self._newimg = self._img

        self._processor = AsyncFunction(processImageAsync, self._img)
        self._processor.finished.connect(self.drawImage)

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
            self.updateImage()
            event.accept()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        event.accept()
        self.zoom(event.delta() > 0)

    def closeEvent(self, event):
        if (self._saved == None):
            event.accept()
        else:
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
                    event.accept()
                else:
                    event.ignore()
            else:
                event.ignore()
        if event.isAccepted():
            super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._saved = None
            self.close()
        elif event.key() == Qt.Key_Q:
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
            self._pic.setPixmap(self._pixmap)
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
            self._pic.setPixmap(self._newpixmap)
        else:
            super().keyReleaseEvent(event)

    def saveImage(self):
        candidate, ext = os.path.splitext(self._filename)
        candidate = candidate + _filename_fmtstring + '.png'
        candidate = candidate.format(self._k_enh, self._σ_enh, self._σ_noise)
        namefilter = 'Image files (*.png *.tiff)'
        filename = QFileDialog.getSaveFileName(self, "Save Image", candidate, namefilter)
        if len(filename) > 0:
            error = ''
            try:
                saveImage(self._newimg, filename)
            except Exception as e:
                error = 'Unexpected error: ' + str(e)
            if len(error) > 0:
                QMessageBox.warning(self, 'Error saving image', error)
                return False
            self._saved = True
            return True

    def updateImage(self):
        if not self._processor.isRunning():
            self._processor.apply(self._k_enh, self._σ_enh, self._σ_noise)
            self._overlay.showBusy(True)
        else:
            self._doUpdate = True

    def drawImage(self):
        self._newimg, pixBarr = self._processor.result()
        if self._doUpdate:
            self._doUpdate = False
            self.updateImage()
        else:
            self._overlay.showBusy(False)
        s = QDataStream(pixBarr)
        pix = QPixmap()
        s.__rshift__(pix)
        self._pic.setPixmap(pix)
        self._newpixmap = pix

    def zoom(self, what):
        if what == None:
            self.setTransform(QTransform())
        elif what == True:
            self.scale(_zoom, _zoom)
        elif what == False:
            self.scale(1./_zoom, 1./_zoom)

    _k_enh = _k_enh_default
    _σ_enh = _σ_enh_default
    _σ_noise = _σ_noise_default
    _exp_factor = 200.
    _lastPos = QPoint()
    _doUpdate = False
    _saved = True
    _doNotOperate = False

if __name__ == '__main__':

    # Check if any arguments are given. If not, or if a single argument
    # is given, show GUI
    if len(sys.argv) < 2 or (len(sys.argv) < 3 and sys.argv[1] != '-h'):
        app = QApplication(sys.argv)
        enh = None
        try:
            if len(sys.argv) == 2:
                enh = ImageEnhancer(sys.argv[1])
            else:
                namefilter = 'Image files (*.png *.tiff *.ppm *.pnm *.pgm *.jpg *.bmp)'
                filename = QFileDialog.getOpenFileName(None, "Open file", '', namefilter)
                if filename != '':
                    enh = ImageEnhancer(filename)
        except Exception as e:
            error = 'Unexpected error: ' + str(e)
            QMessageBox.warning(None, 'Error saving image', error)
            sys.exit(1)
        if enh != None:
            enh.show()
            retval = app.exec_()
            del enh
            del app
            sys.exit(retval)
        else:
            sys.exit(1)

    # Parse arguments and process the given image.

    def nonneg(arg):
        f = float(arg)
        if f >= 0: return f
        msg = 'Negative image enhancement parameters are not valid.'
        raise ArgumentTypeError(msg)

    description="""
    Enahance image using wavelets. Run without arguments or with a
    single argument giving the input image to show the graphical interface.
    """
    epilog="""
    If the output directory is specified, the processed images are put there,
    otherwise they are put in the same directory as the source image. They are
    named the same as the source image with the values of processing parameters
    appended to their names. This behaviour can be disabled with the '-r' option.
    """
    parser = ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-k', type=nonneg, required=True,
                        help='Enhancement k coefficient.')
    parser.add_argument('-s', type=nonneg, required=True,
                        help='Enhancement σ parameter.')
    parser.add_argument('-n', type=nonneg, required=True,
                        help='Noise σ parameter.')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output directory.')
    parser.add_argument('-r', action='store_true', required=False, default=False,
                        help='Do not rename images.')
    parser.add_argument('-f', action='store_true', required=False, default=False,
                        help='Force processing even if it would overwrite source images.')
    parser.add_argument('images', type=str, nargs='+',
                        help='Images to be processed.')

    args = parser.parse_args(sys.argv[1:])
    if args.r and args.o == None and not args.f:
        print('Renaming disabled but output directory not specified!')
        print('This would overwrite source images. If you are sure you')
        print("want to do this, use the '-f' option.")
        sys.exit(1)

    def batchProcess(filename):
        def rename(name):
            destfile, ext = os.path.splitext(name)
            destfile = destfile + _filename_fmtstring + ext
            destfile = destfile.format(args.k, args.s, args.n)
            return destfile

        if args.o == None:
            if args.r:
                destfile = filename
            else:
                destfile = rename(filename)
        else:
            destfile = os.path.basename(filename)
            if not args.r:
                destfile = rename(destfile)
            destfile = args.o + '/' + destfile
        img, depth = loadImage(filename)
        img = processImage(img, args.k, args.s, args.n)
        saveImage(img, destfile)

    pool = Pool()
    pool.map(batchProcess, args.images)
    pool.close()
    pool.join()
    sys.exit(0)
