__author__ = 'cimple'
from PySide2 import QtGui
from PySide2 import QtCore
import maya.OpenMaya as om
import maya.OpenMayaUI as mui
import maya.cmds as mc
import shiboken


def getMayaWindow():
    pointer = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(pointer), QtGui.QWidget)


class ImgUI():
    def __init__(self):

        ##ui
        self.parent = getMayaWindow()
        self.window = QtGui.QMainWindow(self.parent)
        self.widget = QtGui.QWidget()
        self.window.setCentralWidget(self.widget)
        self.layout = QtGui.QVBoxLayout(self.widget)

        self.imgLabel = QtGui.QLabel("<img src = '/home/d10441/PycharmProjects/DirectFacialRetargeting/example/imgseq/faceSeq000.png'>")
        self.frameSlider = QtGui.QSlider()
        self.frameSlider.setOrientation(QtCore.Qt.Horizontal)
        self.frameSpinBox = QtGui.QSpinBox()
        self.startLE = QtGui.QLineEdit()
        self.endLE = QtGui.QLineEdit()

        self.startLE.setMaximumWidth(40)
        self.endLE.setMaximumWidth(40)
        self.startLE.setText('1')
        self.endLE.setText('730')
        self.frameSlider.setMinimum(int(self.startLE.text()))
        self.frameSlider.setMaximum(int(self.endLE.text()))
        self.frameSpinBox.setMinimum(int(self.startLE.text()))
        self.frameSpinBox.setMaximum(int(self.endLE.text()))
        self.startFrame = 1
        self.endFrame = 730

        ##layout
        self.layout.addWidget(self.imgLabel)
        self.sliderLayout = QtGui.QHBoxLayout()
        self.layout.addLayout(self.sliderLayout)
        self.sliderLayout.addWidget(self.startLE)
        self.sliderLayout.addWidget(self.frameSlider)
        self.sliderLayout.addWidget(self.frameSpinBox)
        self.sliderLayout.addWidget(self.endLE)

        ##connect
        self.frameSlider.valueChanged.connect(self.frameSpinBox.setValue)
        self.frameSpinBox.valueChanged.connect(self.frameSlider.setValue)
        self.frameSlider.valueChanged.connect(self.frameSliderChanged)
        self.startLE.textChanged.connect(self.startLEchanged)
        self.endLE.textChanged.connect(self.endLEchanged)

        self.frameCallback = om.MEventMessage.addEventCallback('timeChanged', self.mayaTimeSliderChanged)


    def mayaTimeSliderChanged(self, msg):
        currentTime = int(mc.currentTime(q=True))
        self.frameSlider.setValue(self.startFrame+currentTime-1)

    def startLEchanged(self):
        startVal = int(self.startLE.text())
        self.frameSlider.setMinimum(startVal)
        self.frameSpinBox.setMinimum(int(self.startLE.text()))
        self.startFrame = startVal

    def endLEchanged(self):
        endVal = int(self.endLE.text())
        self.frameSlider.setMaximum(endVal)
        self.frameSpinBox.setMaximum(int(self.endLE.text()))
        self.endFrame = endVal


    def frameSliderChanged(self):
        value = self.frameSlider.value()
        if value < 10 :
            self.imgLabel.setText("<img src = '/home/d10441/PycharmProjects/DirectFacialRetargeting/example/imgseq/faceSeq00"+str(value)+".png'>")
        elif value >= 10 and value < 100 :
            self.imgLabel.setText("<img src = '/home/d10441/PycharmProjects/DirectFacialRetargeting/example/imgseq/faceSeq0"+str(value)+".png'>")
        elif value >= 100 :
            self.imgLabel.setText("<img src = '/home/d10441/PycharmProjects/DirectFacialRetargeting/example/imgseq/faceSeq"+str(value)+".png'>")

    def show(self):
        self.window.show()

    def close(self):
        self.window.close()
        om.MEventMessage.removeCallback(self.frameCallback)



img = ImgUI()
img.show()
#img.close()
