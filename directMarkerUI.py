from __future__ import division
#from PySide import QtCore, QtGui, QtUiTools
#import shiboken
import shiboken2
from shiboken2 import wrapInstance
from PySide2 import QtCore, QtGui, QtUiTools, QtWidgets
import maya.OpenMayaUI as apiUI
import os
import pickle

import maya.cmds as mc
import numpy as np
import sys
currentPath = '/home/d10441/PycharmProjects/DirectFacialRetargeting'
sys.path.append(currentPath)
import CFR_training
reload(CFR_training)
from CFR_training import *
dot = np.dot
DEBUG=0

#...............................................................................
class PCA:
    def __init__( self, A, fraction=0.90 ):
        assert 0 <= fraction <= 1
            # A = U . diag(d) . Vt, O( m n^2 ), lapack_lite --
        self.U, self.d, self.Vt = np.linalg.svd( A, full_matrices=False )
        assert np.all( self.d[:-1] >= self.d[1:] )  # sorted
        self.eigen = self.d**2
        self.sumvariance = np.cumsum(self.eigen)
        self.sumvariance /= self.sumvariance[-1]
        self.npc = np.searchsorted( self.sumvariance, fraction ) + 1
        self.dinv = np.array([ 1/d if d > self.d[0] * 1e-6  else 0
                                for d in self.d ])

    def pc( self ):
        """ e.g. 1000 x 2 U[:, :npc] * d[:npc], to plot etc. """
        n = self.npc
        return self.U[:, :n] * self.d[:n]

    # These 1-line methods may not be worth the bother;
    # then use U d Vt directly --

    def vars_pc( self, x ):
        n = self.npc
        return self.d[:n] * dot( self.Vt[:n], x.T ).T  # 20 vars -> 2 principal

    def pc_vars( self, p ):
        n = self.npc
        return dot( self.Vt[:n].T, (self.dinv[:n] * p).T ) .T  # 2 PC -> 20 vars

    def pc_obs( self, p ):
        n = self.npc
        return dot( self.U[:, :n], p.T )  # 2 principal -> 1000 obs

    def obs_pc( self, obs ):
        n = self.npc
        return dot( self.U[:, :n].T, obs ) .T  # 1000 obs -> 2 principal

    def obs( self, x ):
        return self.pc_obs( self.vars_pc(x) )  # 20 vars -> 2 principal -> 1000 obs

    def vars( self, obs ):
        return self.pc_vars( self.obs_pc(obs) )  # 1000 obs -> 2 principal -> 20 vars


class Center:
    """ A -= A.mean() /= A.std(), inplace -- use A.copy() if need be
        uncenter(x) == original A . x
    """
        # mttiw
    def __init__( self, A, axis=0, scale=True, verbose=1 ):
        self.mean = A.mean(axis=axis)
        if verbose:
            if DEBUG==1 : print "Center -= A.mean:", self.mean
        A -= self.mean
        if scale:
            std = A.std(axis=axis)
            self.std = np.where( std, std, 1. )
            if verbose:
                if DEBUG==1 : print "Center /= A.std:", self.std
            A /= self.std
        else:
            self.std = np.ones( A.shape[-1] )
        self.A = A

    def uncenter( self, x ):
        return np.dot( self.A, x * self.std ) + np.dot( x, self.mean )

class MarkerData():
    def __init__(self, markerList, frameRange):
        self.markerList = markerList
        self.frameRange = frameRange
        self.markerDataList = []
        #self.getMarkerData()
        self.markerTrainDataList = []

    def addMarkerTrainData(self, frame):
        self.markerTrainDataList.append(self.markerDataList[frame-1])

    def removeMarkerTrainData(self, idx):
        self.markerTrainDataList.pop(idx)

    def getMarkerData(self):
        rawMarkerData = []
        for t in range(self.frameRange[0], self.frameRange[1]+1) :
            tmpList = []
            for marker in self.markerList :
                tmpList = tmpList + list(mc.getAttr(marker+'.translate', t=t)[0])
                #mc.currentTime(t)
                #tmpList = tmpList + mc.xform(marker, q=True, t=True, ws=True)
            rawMarkerData.append(tmpList)
        rawMarkerData = np.array(rawMarkerData)
        Center(rawMarkerData)
        fraction = .95
        p = PCA(rawMarkerData, fraction=fraction)
        self.markerDataList = p.pc()


class FaceRig():
    def __init__(self, ctrlList):
        self.ctrlList = ctrlList
        self.ctrlAttrList = []
        self.rigTrainDataList = []
        self.getCtrlAttrList()

    def getCtrlAttrList(self):
        for ctrl in self.ctrlList :
            attrList = mc.listAttr(ctrl, k=True)
            for attr in attrList :
                self.ctrlAttrList.append(ctrl+'.'+attr)

    def setCtrlAttrList(self, ctrlAttrValList):
        for i, ctrlAttr in enumerate(self.ctrlAttrList):
            mc.setAttr(ctrlAttr, ctrlAttrValList[i])
            mc.setKeyframe(ctrlAttr)

    def addRigTrainData(self):
        tmpList = []
        for ctrlAttr in self.ctrlAttrList :
            tmpList.append(mc.getAttr(ctrlAttr))
            mc.setKeyframe(ctrlAttr)
        self.rigTrainDataList.append(tmpList)

    def removeRigTrainData(self, idx):
        for ctr in self.ctrlAttrList :
            f = mc.currentTime(q=True)
            mc.cutKey(ctr, t=(f,f))
        self.rigTrainDataList.pop(idx)

    def editRigTrainData(self, idx):
        tmpList = []
        for ctrlAttr in self.ctrlAttrList :
            tmpList.append(mc.getAttr(ctrlAttr))
            mc.setKeyframe(ctrlAttr)
        self.rigTrainDataList[idx] = tmpList



class CFR_MarkerDirect():
    def __init__(self, markerList, faceRigCtrlList, frameRange):
        self.marker = MarkerData(markerList, frameRange)
        self.faceRig = FaceRig(faceRigCtrlList)
        self.trainingFrameList = []

    def editTrainData(self, frame):
        idx = self.trainingFrameList.index(frame)
        self.faceRig.editRigTrainData(idx)

    def addTrainData(self):
        self.marker.addMarkerTrainData(mc.currentTime(q=True))
        self.faceRig.addRigTrainData()
        self.trainingFrameList.append(mc.currentTime(q=True))

    def removeTrainData(self, idx):
        self.marker.removeMarkerTrainData(idx)
        self.faceRig.removeRigTrainData(idx)
        self.trainingFrameList.pop(idx)


    def running(self):
        """
        RBF Training
        """
        rbfTrain = RBFtrain()
        markerDataArr = np.array(self.marker.markerTrainDataList)
        ctrlDataArr = np.array(self.faceRig.rigTrainDataList)
        rbfTrain.RBFtraining(markerDataArr, ctrlDataArr)
        result = rbfTrain.RBFrunning(self.marker.markerDataList)
        for f, data in enumerate(result):
            #mc.currentTime(f)
            for i, ctrlAttr in enumerate(self.faceRig.ctrlAttrList) :
                mc.setAttr(ctrlAttr, data[i])
                mc.setKeyframe(ctrlAttr, t=f+1)

class DirectRetargetingWidget(QtWidgets.QFrame):
    def __init__(self):
        QtWidgets.QFrame.__init__(self)
        self.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Raised)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.widgetuiFilePath = currentPath+'/ui/directFacialRetargetingWidget.ui'
        self.widgetUI = self.loadUiWidget(self.widgetuiFilePath)
        self.styleFilePath = currentPath+'/stylesheet/widgetScheme.qss'
        with open(self.styleFilePath, 'r') as file:
            style_sheet = file.read()
        self.setStyleSheet(style_sheet)
        self.Instance = None
        self.MarkerList = []
        self.CtrlList = []
        self.connectSignals()


    def loadUiWidget(self, uifilename):
        loader = QtUiTools.QUiLoader()
        uifile = QtCore.QFile(uifilename)
        uifile.open(QtCore.QFile.ReadOnly)
        ui = loader.load(uifile, self)
        uifile.close()
        return ui

    def connectSignals(self):
        self.widgetUI.ui_MarkerSetBtn.clicked.connect(self.ui_MarkerSetBtnClicked)
        self.widgetUI.ui_MarkerResetBtn.clicked.connect(self.ui_MarkerResetBtnClicked)
        self.widgetUI.ui_CtrlSetBtn.clicked.connect(self.ui_CtrlSetBtnClicked)
        self.widgetUI.ui_CtrlResetBtn.clicked.connect(self.ui_CtrlResetBtnClicked)
        self.widgetUI.ui_ReadyForTrainingBtn.clicked.connect(self.ui_ReadyForTrainingBtnClicked)
        self.widgetUI.ui_TrainingAddBtn.clicked.connect(self.ui_TrainingAddBtnClicked)
        self.widgetUI.ui_TrainingSlider.valueChanged.connect(self.ui_TrainingSliderValueChanged)
        self.widgetUI.ui_TrainingRemoveBtn.clicked.connect(self.ui_TrainingRemoveBtnClicked)
        self.widgetUI.ui_TrainingSetNeutralBtn.clicked.connect(self.ui_TrainingSetNeutralBtnClicked)
        self.widgetUI.ui_RetargetingBtn.clicked.connect(self.ui_RetargetingBtnClicked)
        self.widgetUI.ui_markerListWidget.itemClicked.connect(self.ui_markerListWidgetItemClicked)
        self.widgetUI.ui_ctrlListWidget.itemClicked.connect(self.ui_ctrlListWidgetItemClicked)

    def ui_ctrlListWidgetItemClicked(self, item):
        mc.select(item.text(), r=True)

    def ui_markerListWidgetItemClicked(self, item):
        mc.select(item.text(), r=True)

    def ui_RetargetingBtnClicked(self):
        self.Instance.running()

    def ui_TrainingSetNeutralBtnClicked(self):
        attrList = self.Instance.faceRig.ctrlAttrList
        neutralValList = self.Instance.faceRig.rigTrainDataList[0]
        for i, attr in enumerate(attrList):
            mc.setAttr(attr, neutralValList[i])

    def ui_TrainingSliderValueChanged(self, val):
        if len(self.Instance.trainingFrameList)>0 :
            mc.currentTime(self.Instance.trainingFrameList[val-1])
        else:
            return
        #print self.Instance.faceRig.rigTrainDataList


    def ui_MarkerSetBtnClicked(self):
        self.MarkerList = mc.ls(sl=True)
        for marker in self.MarkerList :
            self.widgetUI.ui_markerListWidget.addItem(marker)

    def ui_MarkerResetBtnClicked(self):
        self.MarkerList = []
        self.widgetUI.ui_markerListWidget.clear()

    def ui_CtrlSetBtnClicked(self):
        self.CtrlList = mc.ls(sl=True)
        for ctrl in self.CtrlList :
            self.widgetUI.ui_ctrlListWidget.addItem(ctrl)

    def ui_CtrlResetBtnClicked(self):
        self.CtrlList = []
        self.widgetUI.ui_ctrlListWidget.clear()

    def ui_ReadyForTrainingBtnClicked(self):
        if len(self.MarkerList)==0 :
            print 'ERROR: No  Marker List'
            return
        if len(self.CtrlList)==0:
            print 'ERROR: No  Controller List'
            return
        startFrame = self.widgetUI.ui_StartFrameSpinBox.value()
        endFrame = self.widgetUI.ui_EndFrameSpinBox.value()
        if endFrame-startFrame <= 0 :
            print 'ERROR: Invalid Frame Range'
            return
        self.Instance = CFR_MarkerDirect(self.MarkerList, self.CtrlList, [startFrame, endFrame])
        self.Instance.marker.getMarkerData()
        self.ui_trainingPartEnable()

    def ui_TrainingAddBtnClicked(self):
        if mc.currentTime(q=True) in self.Instance.trainingFrameList :
            self.Instance.editTrainData(mc.currentTime(q=True))
        else:
            self.Instance.addTrainData()
            self.widgetUI.ui_TrainingSlider.setMinimum(1)
            self.widgetUI.ui_TrainingSlider.setMaximum(len(self.Instance.trainingFrameList))
            self.widgetUI.ui_TrainingSlider.setValue(len(self.Instance.trainingFrameList))
            self.widgetUI.ui_TrainingSpinBox.setMinimum(1)
            self.widgetUI.ui_TrainingSpinBox.setMaximum(len(self.Instance.trainingFrameList))
            self.widgetUI.ui_TrainingSpinBox.setValue(len(self.Instance.trainingFrameList))
            self.widgetUI.ui_TrainingTotalExampleNumberLabel.setText('/'+str(len(self.Instance.trainingFrameList)))

        if len(self.Instance.trainingFrameList) >= 2:
            self.widgetUI.ui_RetargetingBtn.setEnabled(True)

    def ui_TrainingRemoveBtnClicked(self):
        idx = self.widgetUI.ui_TrainingSlider.value()
        if idx < 1:
            print 'ERORR: Cannot Remove Anything'
        elif idx == 1:
            self.Instance.removeTrainData(idx-1)
            self.widgetUI.ui_TrainingSlider.setMaximum(len(self.Instance.trainingFrameList))
            self.widgetUI.ui_TrainingSpinBox.setMaximum(len(self.Instance.trainingFrameList))
            self.widgetUI.ui_TrainingSlider.setMinimum(0)
            self.widgetUI.ui_TrainingSpinBox.setMinimum(0)
            self.widgetUI.ui_TrainingTotalExampleNumberLabel.setText('/0')
        else:
            self.Instance.removeTrainData(idx-1)
            self.widgetUI.ui_TrainingSlider.setMaximum(len(self.Instance.trainingFrameList))
            self.widgetUI.ui_TrainingSpinBox.setMaximum(len(self.Instance.trainingFrameList))
            self.widgetUI.ui_TrainingTotalExampleNumberLabel.setText('/'+str(len(self.Instance.trainingFrameList)))

        if len(self.Instance.trainingFrameList) <= 1:
            self.widgetUI.ui_RetargetingBtn.setEnabled(False)


    def ui_trainingPartEnable(self):
        self.widgetUI.ui_TrainingTitleLabel.setEnabled(True)
        self.widgetUI.ui_TrainingSlider.setEnabled(True)
        self.widgetUI.ui_TrainingSpinBox.setEnabled(True)
        self.widgetUI.ui_TrainingTotalExampleNumberLabel.setEnabled(True)
        self.widgetUI.ui_TrainingAddBtn.setEnabled(True)
        self.widgetUI.ui_TrainingRemoveBtn.setEnabled(True)
        self.widgetUI.ui_TrainingSetNeutralBtn.setEnabled(True)
        self.widgetUI.ui_TrainingNoticeLabel.setEnabled(True)

    def ui_trainingPartDisable(self):
        self.widgetUI.ui_TrainingTitleLabel.setEnabled(False)
        self.widgetUI.ui_TrainingSlider.setEnabled(False)
        self.widgetUI.ui_TrainingSpinBox.setEnabled(False)
        self.widgetUI.ui_TrainingTotalExampleNumberLabel.setEnabled(False)
        self.widgetUI.ui_TrainingAddBtn.setEnabled(False)
        self.widgetUI.ui_TrainingRemoveBtn.setEnabled(False)
        self.widgetUI.ui_TrainingSetNeutralBtn.setEnabled(False)
        self.widgetUI.ui_TrainingNoticeLabel.setEnabled(False)


class TabDeleteDialog(QtWidgets.QDialog):
    def __init__(self, tabWidget, UIInatance):
        super(TabDeleteDialog, self).__init__(parent=tabWidget)
        global tabDeleteDialog
        try:
            tabDeleteDialog.close()
            tabDeleteDialog.deleteLater()
        except: pass

        self.setWindowTitle('Delete Tab')
        self.tabWidget = tabWidget
        self.UIInatance = UIInatance

        self.warningLabel = QtWidgets.QLabel('Are you sure to delete current tab?')
        self.okBtn = QtWidgets.QPushButton('OK', parent=self)
        self.closeBtn = QtWidgets.QPushButton('Close', parent=self)

        self.layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, self)
        self.layout.addWidget(self.warningLabel)
        self.layout.addWidget(self.okBtn)
        self.layout.addWidget(self.closeBtn)
        self.connections()

    def okBtnClicked(self):
        self.UIInatance.DirectRetargetingWidgetList.pop(self.tabWidget.currentIndex())
        self.tabWidget.removeTab(self.tabWidget.currentIndex())
        self.close()


    def connections(self):
        self.okBtn.clicked.connect(self.okBtnClicked)
        self.closeBtn.clicked.connect(self.close)



class TabRenameDialog(QtWidgets.QDialog):
    def __init__(self, tabWidget):
        super(TabRenameDialog, self).__init__(parent=tabWidget)
        global tabRenameDialog
        try:
            tabRenameDialog.close()
            tabRenameDialog.deleteLater()
        except: pass

        self.setWindowTitle('Rename Tab')
        self.tabWidget = tabWidget

        self.nameLE = QtWidgets.QLineEdit('NewName', parent=self)
        self.renameBtn = QtWidgets.QPushButton('Rename', parent=self)
        self.closeBtn = QtWidgets.QPushButton('Close', parent=self)

        self.layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, self)
        self.layout.addWidget(self.nameLE)
        self.layout.addWidget(self.renameBtn)
        self.layout.addWidget(self.closeBtn)
        self.connections()

    def renameBtnClicked(self):
        self.tabWidget.setTabText(self.tabWidget.currentIndex(), self.nameLE.text())
        self.close()

    def connections(self):
        self.renameBtn.clicked.connect(self.renameBtnClicked)
        self.closeBtn.clicked.connect(self.close)



class DirectRetargetingWindow(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        self.uiFilePath = currentPath+'/ui/tabWidget.ui'
        self.MainWindow = None
        self.setObjectName('DirectFacialRetargetingUI')
        self.DirectRetargetingWidgetList = []
        self._dock_widget = self._dock_name = None
        self.setLayout(QtWidgets.QVBoxLayout())
        self.styleFilePath = currentPath+'/stylesheet/dockUIScheme.qss'
        #self.styleFilePath.open(QtCore.QFile.ReadOnly)
        with open(self.styleFilePath, 'r') as file:
            style_sheet = file.read()
        self.setStyleSheet(style_sheet)


    def getMayaWindow():
        ptr = apiUI.MQtUtil.mainWindow()
        if ptr is not None:
            return shiboken2.wrapInstance(long(ptr), QtWidgets.QWidget)


    def loadUiWidget(self, uifilename, parent=getMayaWindow()):
        loader = QtUiTools.QUiLoader()
        uifile = QtCore.QFile(uifilename)
        uifile.open(QtCore.QFile.ReadOnly)
        ui = loader.load(uifile, parent)
        uifile.close()
        return ui

    def connectSignals(self):
        self.MainWindow.ui_LoadTrainingDataBtn.clicked.connect(self.ui_LoadTrainingDataBtnClicked)
        self.MainWindow.ui_SaveTrainingDataBtn.clicked.connect(self.ui_SaveTrainingDataBtnClicked)
        self.MainWindow.ui_CreateNewTabBtn.clicked.connect(self.ui_CreateNewTabBtnClicked)
        self.MainWindow.tabWidget.customContextMenuRequested.connect(self.tabWidgetContextMenu)


    def tabWidgetContextMenu(self, pos):
        if self.MainWindow.tabWidget.currentWidget() == None :
            return
        contextMenu = QtWidgets.QMenu(self)
        action1 = contextMenu.addAction('Rename Tab')
        action2 = contextMenu.addAction('Delete Tab')
        action1.triggered.connect(self.popupAction_renameTab)
        action2.triggered.connect(self.popupAction_deleteTab)
        contextMenu.exec_(self.MainWindow.tabWidget.currentWidget().mapToGlobal(pos))


    def popupAction_renameTab(self):
        renameDialog = TabRenameDialog(self.MainWindow.tabWidget)
        renameDialog.show()

    def popupAction_deleteTab(self):
        deleteTabDialog = TabDeleteDialog(self.MainWindow.tabWidget, self)
        deleteTabDialog.show()

    def ui_CreateNewTabBtnClicked(self):
        new_widget = DirectRetargetingWidget()
        self.DirectRetargetingWidgetList.append(new_widget)
        tab = QtWidgets.QWidget()
        vLayout = QtWidgets.QVBoxLayout(tab)
        vLayout.setContentsMargins(5,5,5,5)
        vLayout.addWidget(new_widget)
        self.MainWindow.tabWidget.addTab(tab, self.MainWindow.ui_CharacterNameLE.text())
        self.MainWindow.tabWidget.setCurrentIndex(self.MainWindow.tabWidget.count()-1)

    def ui_SaveTrainingDataBtnClicked(self):
        saveInstance = SaveLoadFileData()
        for i in range(self.MainWindow.tabWidget.count()):
            saveInstance.tabList.append(self.MainWindow.tabWidget.tabText(i))

        #print 'MarkeRList : ', self.DirectRetargetingWidgetList[0].MarkerList

        for i, DirectRetargetingWidget in enumerate(self.DirectRetargetingWidgetList) :
            saveInstance.markerListDic[saveInstance.tabList[i]] = DirectRetargetingWidget.MarkerList
            saveInstance.ctrlListDic[saveInstance.tabList[i]] = DirectRetargetingWidget.CtrlList
            saveInstance.trainingFrameListDic[saveInstance.tabList[i]] = DirectRetargetingWidget.Instance.trainingFrameList
            saveInstance.frameRangeDic[saveInstance.tabList[i]] = [DirectRetargetingWidget.widgetUI.ui_StartFrameSpinBox.value(), DirectRetargetingWidget.widgetUI.ui_EndFrameSpinBox.value()]
            saveInstance.markerDataListDic[saveInstance.tabList[i]] = DirectRetargetingWidget.Instance.marker.markerTrainDataList
            saveInstance.rigTrainingDataListDic[saveInstance.tabList[i]] = DirectRetargetingWidget.Instance.faceRig.rigTrainDataList

        filePath = mc.fileDialog2(okCaption='Data Save', ff='*.txt', fm=0)
        f = open(filePath[0], 'w')
        pickle.dump(saveInstance, f)
        f.close()

    def ui_LoadTrainingDataBtnClicked(self):
        filePath = mc.fileDialog2(okCaption='Data Load', ff='*.txt', fm=1)[0]
        self.loadTrainingFile(filePath)

    def loadTrainingFile(self, filePath):
        f = open(filePath)
        loadInstance = pickle.load(f)
        f.close()

        tabList = loadInstance.tabList

        for tabName in tabList :
            new_widget = DirectRetargetingWidget()
            markerList = loadInstance.markerListDic[tabName]
            ctrlList = loadInstance.ctrlListDic[tabName]
            trainingFrameList = loadInstance.trainingFrameListDic[tabName]
            markerDataList = loadInstance.markerDataListDic[tabName]
            rigTrainDataList = loadInstance.rigTrainingDataListDic[tabName]
            frameRange = loadInstance.frameRangeDic[tabName]

            new_widget.Instance = CFR_MarkerDirect(markerList, ctrlList, [frameRange[0], frameRange[1]])
            new_widget.Instance.marker.frameRange = frameRange
            new_widget.Instance.marker.markerList = markerList
            new_widget.Instance.marker.getMarkerData()
            #new_widget.Instance.marker.markerDataList = markerDataList
            new_widget.Instance.marker.markerTrainDataList = markerDataList
            new_widget.Instance.faceRig.rigTrainDataList = rigTrainDataList
            new_widget.Instance.trainingFrameList = trainingFrameList

            for marker in markerList :
                new_widget.widgetUI.ui_markerListWidget.addItem(marker)
            for ctrl in ctrlList :
                new_widget.widgetUI.ui_ctrlListWidget.addItem(ctrl)

            new_widget.widgetUI.ui_StartFrameSpinBox.setValue(frameRange[0])
            new_widget.widgetUI.ui_EndFrameSpinBox.setValue(frameRange[1])
            new_widget.ui_trainingPartEnable()
            new_widget.widgetUI.ui_TrainingSlider.setMinimum(1)
            new_widget.widgetUI.ui_TrainingSlider.setMaximum(len(new_widget.Instance.trainingFrameList))
            new_widget.widgetUI.ui_TrainingSlider.setValue(len(new_widget.Instance.trainingFrameList))
            new_widget.widgetUI.ui_TrainingSpinBox.setMinimum(1)
            new_widget.widgetUI.ui_TrainingSpinBox.setMaximum(len(new_widget.Instance.trainingFrameList))
            new_widget.widgetUI.ui_TrainingSpinBox.setValue(len(new_widget.Instance.trainingFrameList))
            new_widget.widgetUI.ui_TrainingTotalExampleNumberLabel.setText('/'+str(len(new_widget.Instance.trainingFrameList)))

            if len(new_widget.Instance.trainingFrameList) >= 2:
                new_widget.widgetUI.ui_RetargetingBtn.setEnabled(True)


            self.DirectRetargetingWidgetList.append(new_widget)

            tab = QtWidgets.QWidget()
            vLayout = QtWidgets.QVBoxLayout(tab)
            vLayout.setContentsMargins(5,5,5,5)
            vLayout.addWidget(new_widget)
            self.MainWindow.tabWidget.addTab(tab, tabName)
            self.MainWindow.tabWidget.setCurrentIndex(self.MainWindow.tabWidget.count()-1)


    def connectDockWidget(self, dock_name, dock_widget):
        self._dock_widget = dock_widget
        self._dock_name = dock_name

    def close(self):
        if self._dock_widget:
            mc.deleteUI(self._dock_name)
        else:
            QtWidgets.QDialog.close(self)
        self._dock_widget = self._dock_name = None

class SaveLoadFileData():
    def __init__(self):
        self.tabList = []
        self.markerListDic = {}
        self.ctrlListDic = {}
        self.trainingFrameListDic = {}
        self.markerDataListDic = {}
        self.rigTrainingDataListDic = {}
        self.frameRangeDic = {}


def main(docked=True):
    global PyForm
    PyForm=DirectRetargetingWindow()

    if docked is True:
        PyForm.close()
        ptr = apiUI.MQtUtil.mainWindow()
        if ptr is not None:
            main_window = shiboken2.wrapInstance(long(ptr), QtWidgets.QWidget)
        PyForm.MainWindow = PyForm.loadUiWidget(PyForm.uiFilePath)
        PyForm.layout().addWidget(PyForm.MainWindow)
        PyForm.connectSignals()
        PyForm.setParent(main_window)
        size = PyForm.size()

        name = apiUI.MQtUtil.fullName(long(shiboken2.getCppPointer(PyForm)[0]))
        dock = mc.dockControl(
            allowedArea = ['right', 'left'],
            area = 'right',
            floating = False,
            content = name,
            width = size.width(),
            height = size.height(),
            label = 'Direct Facial Retargeting UI')

        widget = apiUI.MQtUtil.findControl(dock)
        dock_widget = shiboken2.wrapInstance(long(widget), QtCore.QObject)
        PyForm.connectDockWidget(dock, dock_widget)
    PyForm.show()