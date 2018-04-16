'''
Created on 2015. 6. 1.

@author: cimple
'''

import numpy as np
CFR_trainig_DEBUG = 0

class RBF :
    def __init__(self) :
        self.reset()        
    def __del__(self) :
        self.reset()
    def reset(self) :
        self.basisType = 'HARDY'        
        self.lamda = 0.0
        self.numInput = 0
        self.dimInput = 0
        self.dimOutput = 0
        self.inputMat = np.arange(0.0)
        self.basisMat = np.arange(0.0)
        self.inverseBasisMat = np.arange(0.0)
        self.weightMat = np.arange(0.0)
        self.minDist = np.arange(0.0)
        return;
    
    def buildDistMatrix(self, distMat, inputMat):        
        for i in range(self.numInput) :
            for j in range(self.numInput) :
                if(i == j) : distMat[i,j] = 0.0
                else : distMat[i,j] = self.dist(inputMat[i], inputMat[j])
        for i in range(self.numInput) :
            for j in range(self.numInput) :
                dmin = float("inf")
                if distMat[i,j] < dmin :
                    dmin = distMat[i, j]                    
                    self.minDist[i] = dmin
        return;
    
    def basisFunc(self, i, x2) :
        if self.basisType == 'HARDY' :
            return (x2 + self.minDist[i])**0.5
        elif self.basisType == 'GAUSSIAN' :
            return np.exp(-x2*0.1*0.1)
        elif self.basisType == 'LINEAR' :
            return abs((x2)**0.5)            
        else :
            print "Wrong basis type!!"
            return;
            
    def dist(self, a, b) :
        d1 = 0.0
        d2 = 0.0

        dim = a.shape[0]

        for i in range(dim) :
            d1 = a[i] - b[i]
            d2 += (d1*d1)

        return d2                
        
    def buildBasisMatrix(self, inputMat) :
        self.numInput = inputMat.shape[0]
        if(self.numInput <= 0) : return;
        self.dimInput = inputMat.shape[1]
        distMat = np.zeros((self.numInput,self.numInput))
        self.minDist = np.zeros(self.numInput)
               
        self.buildDistMatrix(distMat, inputMat)
        if CFR_trainig_DEBUG : print "Build dist matrix done"
        self.basisMat = np.zeros((self.numInput, self.numInput))
        self.inverseBasisMat = np.zeros((self.numInput, self.numInput))
        for i in range(self.numInput) :
            for j in range(self.numInput) :               
                self.basisMat[i, j] = self.basisFunc(j, distMat[i, j])
                if(i==j) : self.basisMat[i, j] += self.lamda
        if CFR_trainig_DEBUG : print "Calculate basis mat done"
        self.inverseBasisMat = np.linalg.inv(self.basisMat)
        if CFR_trainig_DEBUG : print "Calculate inverse mat done"
        return;
    
    def setBasisFunction(self, basisType) :
        if basisType not in ['GAUSSIAN', 'HARDY', 'LINEAR'] :
            print "Wrong basis function type!"
            return;
        else :
            self.basisType = basisType
            return;
        
    def setLamda(self, value) :
        self.lamda = value
        return;
    
    def train(self, inputMat, outputMat):
        if(outputMat.shape[0] <= 0) : return;
        if CFR_trainig_DEBUG : print "Build basis matrix start"
        self.dimOutput = outputMat.shape[1]
        self.inputMat = inputMat
        self.buildBasisMatrix(inputMat)
        resultMat = np.zeros((self.numInput, self.dimOutput))
        for i in  range(self.numInput) :
            for j in range(self.dimOutput) :
                resultMat[i,j] = outputMat[i,j]
        self.weightMat = np.dot(self.inverseBasisMat, resultMat)
        return;
    
    def interpolate(self, sample) :
        sampleMat = np.zeros(self.numInput)
        resultMat = np.zeros(self.dimOutput)        
        for i in range(self.numInput) :
            sampleMat[i] = self.basisFunc(i, self.dist(sample, self.inputMat[i]))        
        resultMat = np.dot(sampleMat, self.weightMat)
        result = np.zeros(self.dimOutput)
        
        for i in range(self.dimOutput) :            
            result[i] = resultMat[i]
        return result;  
    
    
class RBFtrain :    
    def __init__(self):
        self.rbfn = None
        
    def RBFtraining(self, srcROE, tgtROE) :
        if srcROE.shape[0] != tgtROE.shape[0] :
            print "ERROR: ROE size of source and target is different!"
            return        
        self.rbfn = RBF()
        self.rbfn.setBasisFunction('HARDY')
        self.rbfn.setLamda(0.1)
        self.rbfn.train(srcROE, tgtROE)
    
    def RBFrunning(self, srcAnimDataList):
        resultAnimDataList = []
        for srcAnimData in srcAnimDataList :
            result = self.rbfn.interpolate(srcAnimData)
            resultAnimDataList.append(result)
        return resultAnimDataList
        
        
#        
#    def RBF_interpolate(self, srcInput, trainData, numTargetCV):
#        rSrcInput = np.zeros(trainData.rbfTrain.rbfn.numInput)
#        for i in range(len(rSrcInput)) :
#            rSrcInput[i] = self.basisFunc(i, self.dist(srcInput, trainData.rbfTrain.rbfn.inputMat[i]))
#        resultMat = np.dot(rSrcInput, trainData.weightMat)
#        result = np.zeros(numTargetCV)        
#        for i in range(numTargetCV) :            
#            result[i] = resultMat[i]        
#        return result
#    
#    def CFR_running(self, srcAnimData, tgtCharData, trainData):
#        numTargetCV = tgtCharData.numCV
#        resultAnimDataList = []
#        for i in range(len(srcAnimData.animDataList)):
#            out = self.RBF_interpolate(srcAnimData.animDataList[i], trainData, numTargetCV)
#            resultAnimDataList.append(out)
#        return resultAnimDataList
          
        