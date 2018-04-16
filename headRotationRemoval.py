"""
Head rotation removal Maya python script

Usage:

totalMarkerList = ['marker1', 'marker2', ..., 'marker100']   # create total facial marker list
staticMarkerList = ['marker1', 'marker2', 'marker3']        # create static facial marker list
HR = HeadRotationRemoval(totalMarkerList, staticMarkerList, 1)   # create HR instance with (totalMarker, staticMarker, referenceFrame)
HR.rotationRemoval([1,3066])    # run rotation removal with time range

"""

import maya.cmds as mc
import numpy as np
DEBUG = 0

class HeadRotationRemoval():
    def __init__(self, totalMarkerList, staticMarkerList, referenceFrame):
        self.totalMarkerList = totalMarkerList
        self.staticMarkerList = staticMarkerList
        self.referenceFrame = referenceFrame

    def get_standardized_matrix(self, mat):
        mean = mat.mean(axis=0)
        return mat - mean

    def get_fixed_rotation_matrix(self, initStaticMarkerPosMat, staticMarkerPosMat):
        mean = initStaticMarkerPosMat.mean(axis=0)
        initStaticMarkerPosMat_s = self.get_standardized_matrix(initStaticMarkerPosMat).T
        staticMarkerPosMat_s = self.get_standardized_matrix(staticMarkerPosMat).T
        H = np.dot(staticMarkerPosMat_s, initStaticMarkerPosMat_s.T)
        U,s,V = np.linalg.svd(H)
        I = np.identity(V.shape[0])
        I[V.shape[0]-1, V.shape[0]-1] = np.linalg.det(np.dot(V, U.T))
        R = np.dot(np.dot(V.T, I),U.T)
        return R

    def get_fixed_marker_pos_mat(self, initStaticMarkerPosMat, staticMarkerPosMat, totalMarkerPosMat):
        R = self.get_fixed_rotation_matrix(initStaticMarkerPosMat, staticMarkerPosMat)
        fixedTotalMarkerPosMat = np.dot(R, totalMarkerPosMat.T).T
        fixedStaticMarkerPosMat = np.dot(R, staticMarkerPosMat.T).T
        translationVec = initStaticMarkerPosMat.mean(axis=0) - fixedStaticMarkerPosMat.mean(axis=0)
        return fixedTotalMarkerPosMat+translationVec

    def get_locator_pos_list(self, locList):
        locPosList = [mc.pointPosition(loc, w=True) for loc in locList]
        return locPosList

    def get_marker_pos_matrix(self, markerList, frame):
        mc.currentTime(frame)
        posMat = np.asarray(self.get_locator_pos_list(markerList))
        return posMat

    def rotationRemoval(self, frameRange):
        initStaticMarkerPosMat = self.get_marker_pos_matrix(self.staticMarkerList, self.referenceFrame)  # Reference poses of static markers

        for f in range(frameRange[0], frameRange[1]+1):
            mc.currentTime(f)
            staticMarkerPosMat = self.get_marker_pos_matrix(self.staticMarkerList, f)
            totalMarkerPosMat = self.get_marker_pos_matrix(self.totalMarkerList, f)
            fixedTotalMarkerPosMat = self.get_fixed_marker_pos_mat(initStaticMarkerPosMat, staticMarkerPosMat, totalMarkerPosMat)

            for i, marker in enumerate(self.totalMarkerList) :
                if not mc.objExists('new_'+marker):
                    mc.spaceLocator(n='new_'+marker)
                mc.xform('new_'+marker, t=[fixedTotalMarkerPosMat[i][0], fixedTotalMarkerPosMat[i][1], fixedTotalMarkerPosMat[i][2]], ws=True)
                mc.setKeyframe('new_'+marker)


# totalMarkerList = [u'R_ForeHead_T01', u'R_ForeHead_T02', u'C_ForeHead_T', u'L_ForeHead_T02', u'L_ForeHead_T01', u'R_ForeHead_M01', u'R_ForeHead_M02', u'C_ForeHead_M', u'L_ForeHead_M02', u'L_ForeHead_M01', u'R_ForeHead_B01', u'R_ForeHead_B02', u'C_ForeHead_B03', u'L_ForeHead_B02', u'L_ForeHead_B01', u'R_Brow_01', u'R_Brow_02', u'R_Brow_03', u'R_Brow_04', u'R_Brow_05', u'L_Brow_01', u'L_Brow_02', u'L_Brow_03', u'L_Brow_04', u'L_Brow_05', u'C_Brow', u'R_Eye_01', u'R_Eye_02', u'R_Eye_03', u'R_Eye_04', u'R_Eye_05', u'R_Eye_06', u'R_Eye_07', u'L_Eye_01', u'L_Eye_02', u'L_Eye_03', u'L_Eye_04', u'L_Eye_05', u'L_Eye_06', u'L_Eye_07', u'R_Nose_Top', u'C_Nose_Top', u'L_Nose_Top', u'R_Nose_Mid', u'C_Nose_Mid', u'L_Nose_Mid', u'R_Nose_Bot', u'C_Nose_Bot', u'L_Nose_Bot', u'R_Cheek_T01', u'R_Cheek_T02', u'R_Cheek_T03', u'R_Cheek_T04', u'R_Cheek_M01', u'R_Cheek_M02', u'R_Cheek_M03', u'R_Cheek_M04', u'R_Cheek_B01', u'R_Cheek_B02', u'R_Cheek_B03', u'R_Cheek_B04', u'L_Cheek_T01', u'L_Cheek_T02', u'L_Cheek_T03', u'L_Cheek_T04', u'L_Cheek_M01', u'L_Cheek_M02', u'L_Cheek_M03', u'L_Cheek_M04', u'L_Cheek_B01', u'L_Cheek_B02', u'L_Cheek_B03', u'L_Cheek_B04', u'R_Phil_01', u'L_Phil_01', u'R_Mouth_T01', u'R_Mouth_T02', u'R_Mouth_T03', u'C_Mouth_Top', u'L_Mouth_T03', u'L_Mouth_T02', u'L_Mouth_T01', u'R_Mouth_B01', u'R_Mouth_B02', u'C_Mouth_Bot', u'L_Mouth_B02', u'L_Mouth_B01', u'R_Chin_T01', u'R_Chin_T02', u'R_Chin_T03', u'L_Chin_T03', u'L_Chin_T02', u'L_Chin_T01', u'R_Chin_B01', u'R_Chin_B02', u'R_Chin_B03', u'L_Chin_B03', u'L_Chin_B02', u'L_Chin_B01', u'head0', u'head1', u'head2']
# staticMarkerList = [u'head0', u'head1', u'head2']
# HR = HeadRotationRemoval(totalMarkerList, staticMarkerList, 1)
# HR.rotationRemoval([3200,4145])

totalMarkerList = [u'R_ForeHead_T01', u'R_ForeHead_T02', u'C_ForeHead_T', u'L_ForeHead_T02', u'L_ForeHead_T01', u'R_ForeHead_M01', u'R_ForeHead_M02', u'C_ForeHead_M', u'L_ForeHead_M02', u'L_ForeHead_M01', u'R_ForeHead_B01', u'R_ForeHead_B02', u'C_ForeHead_B03', u'L_ForeHead_B02', u'L_ForeHead_B01', u'R_Brow_01', u'R_Brow_02', u'R_Brow_03', u'R_Brow_04', u'R_Brow_05', u'L_Brow_01', u'L_Brow_02', u'L_Brow_03', u'L_Brow_04', u'L_Brow_05', u'C_Brow', u'R_Eye_01', u'R_Eye_02', u'R_Eye_03', u'R_Eye_04', u'R_Eye_05', u'R_Eye_06', u'R_Eye_07', u'L_Eye_01', u'L_Eye_02', u'L_Eye_03', u'L_Eye_04', u'L_Eye_05', u'L_Eye_06', u'L_Eye_07', u'R_Nose_Top', u'C_Nose_Top', u'L_Nose_Top', u'R_Nose_Mid', u'C_Nose_Mid', u'L_Nose_Mid', u'R_Nose_Bot', u'C_Nose_Bot', u'L_Nose_Bot', u'R_Cheek_T01', u'R_Cheek_T02', u'R_Cheek_T03', u'R_Cheek_T04', u'R_Cheek_M01', u'R_Cheek_M02', u'R_Cheek_M03', u'R_Cheek_M04', u'R_Cheek_B01', u'R_Cheek_B02', u'R_Cheek_B03', u'R_Cheek_B04', u'L_Cheek_T01', u'L_Cheek_T02', u'L_Cheek_T03', u'L_Cheek_T04', u'L_Cheek_M01', u'L_Cheek_M02', u'L_Cheek_M03', u'L_Cheek_M04', u'L_Cheek_B01', u'L_Cheek_B02', u'L_Cheek_B03', u'L_Cheek_B04', u'R_Phil_01', u'L_Phil_01', u'R_Mouth_T01', u'R_Mouth_T02', u'R_Mouth_T03', u'C_Mouth_Top', u'L_Mouth_T03', u'L_Mouth_T02', u'L_Mouth_T01', u'R_Mouth_B01', u'R_Mouth_B02', u'C_Mouth_Bot', u'L_Mouth_B02', u'L_Mouth_B01', u'R_Chin_T01', u'R_Chin_T02', u'R_Chin_T03', u'L_Chin_T03', u'L_Chin_T02', u'L_Chin_T01', u'R_Chin_B01', u'R_Chin_B02', u'R_Chin_B03', u'L_Chin_B03', u'L_Chin_B02', u'L_Chin_B01', u'left', u'center', u'right']
staticMarkerList = ['left', 'center', 'right', 'C_Nose_Bot']
HR = HeadRotationRemoval(totalMarkerList, staticMarkerList, 1)
HR.rotationRemoval([1,725])