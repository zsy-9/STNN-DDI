import numpy as np
from sklearn.model_selection import train_test_split
from random import randint
from numpy import random
from cold_start import getnews


DrugNumber = 555
DDINumber = 1318

#Pubchem
def getstructure():
    dataFolder = 'D:\模型们\pytensor2'
    DrugStructureAddr = dataFolder + '\Drugbank_CID_StrucVactor_pubchem_drug-881Matrix.txt'
    fileIn = open(DrugStructureAddr)
    DrugStructureMatrix = []
    line=fileIn.readline()
    while line:
        lineArr = line.strip().split('\t')
        Structure = lineArr[1].strip().split(' ')
        temp = []
        for i in Structure:
            temp.append(float(i))
        DrugStructureMatrix.append(temp)
        line = fileIn.readline()
    #DrugStructureMatrix=np.mat(DrugStructureMatrix)
    return DrugStructureMatrix

class Drugs(object):
    def __init__(self, Drug1, Drug2, Interaction):
        self.Drug1=Drug1
        self.Drug2=Drug2
        self.Interaction=Interaction

#train_DDI,C2_DDI,C3_DDI
def getDDI():
    dataFolder = 'D:\模型们\pytensor2'
    DrugStructureAddr = dataFolder + '\AdverseDrugDrugInteractionDataset.txt'
    fileIn = open(DrugStructureAddr)
    line = fileIn.readline()
    L1=getnews()
    trDDISET=[]
    tenoDDISET=[]
    tennDDISET=[]
    DrugTensor = np.zeros(((DrugNumber, DrugNumber, DDINumber)))
    while line:
        lineArr = line.strip().split('\t')
        lineArr[0] = int(lineArr[0])
        lineArr[1] = int(lineArr[1])
        lineArr[2] = int(lineArr[2])
        DDIElt = Drugs(lineArr[0], lineArr[1], lineArr[2])
        if lineArr[0] not in L1 and lineArr[1] not in L1:
            trDDISET.append(DDIElt)
        if lineArr[0] in L1 and lineArr[1] not in L1:
            tenoDDISET.append(DDIElt)
        if lineArr[0] not in L1 and lineArr[1] in L1:
            tenoDDISET.append(DDIElt)
        if lineArr[0] in L1 and lineArr[1] in L1:
            tennDDISET.append(DDIElt)
        DrugTensor[lineArr[0], lineArr[1], lineArr[2]]=1
        DrugTensor[lineArr[1], lineArr[0], lineArr[2]]=1
        line = fileIn.readline()
    return trDDISET,tenoDDISET,tennDDISET,DrugTensor

#train_DDI,C1_DDI
def getDDI1():
    dataFolder = 'D:\模型们\pytensor2'
    DrugStructureAddr = dataFolder + '\AdverseDrugDrugInteractionDataset.txt'
    fileIn = open(DrugStructureAddr)
    line = fileIn.readline()
    DDISET=[]
    DrugTensor = np.zeros(((DrugNumber, DrugNumber, DDINumber)))
    while line:
        lineArr = line.strip().split('\t')
        lineArr[0] = int(lineArr[0])
        lineArr[1] = int(lineArr[1])
        lineArr[2] = int(lineArr[2])
        DDIElt = Drugs(lineArr[0], lineArr[1], lineArr[2])
        DDISET.append(DDIElt)
        DrugTensor[lineArr[0], lineArr[1], lineArr[2]]=1
        DrugTensor[lineArr[1], lineArr[0], lineArr[2]]=1
        line = fileIn.readline()
    return DDISET,DrugTensor


trDDISET,tenoDDISET,tennDDISET,DrugTensor=getDDI()
DDISET,DrugTensor=getDDI1()
print('DDI get')
#Random arrangement
def gettvt():
    trDDISET1, _ = train_test_split(trDDISET, test_size=0.000001)
    tenoDDISET1, _ = train_test_split(tenoDDISET, test_size=0.000001)
    tennDDISET1, _ = train_test_split(tennDDISET, test_size=0.000001)
    return trDDISET1,tenoDDISET1,tennDDISET1


DrugStructureMatrix=getstructure()
InteractionMartix= random.random(size=(DDINumber,DDINumber))

#Negative sample
def randomgetone():
    temp = []
    temp.append(randint(0, 554))
    temp.append(randint(0, 554))
    temp.append(randint(0, 1317))
    return temp

def getdata(set):
    DDIstructure_1=[]
    DDIstructure_2=[]
    DDIinteraction=[]
    usefulDDI=[]
    labels=[]
    for DDI in set:
        temp=[]
        temp.append(DDI.Drug1)
        DDIstructure_1.append(DrugStructureMatrix[DDI.Drug1])
        temp.append(DDI.Drug2)
        DDIstructure_2.append(DrugStructureMatrix[DDI.Drug2])
        temp.append(DDI.Interaction)
        DDIinteraction.append(InteractionMartix[DDI.Interaction])
        labels.append(1)
        usefulDDI.append(temp)
        randomtemp=randomgetone()
        DDIstructure_1.append(DrugStructureMatrix[randomtemp[0]])
        DDIstructure_2.append(DrugStructureMatrix[randomtemp[1]])
        DDIinteraction.append(InteractionMartix[randomtemp[2]])
        usefulDDI.append(randomtemp)
        labels.append(DrugTensor[randomtemp[0],randomtemp[1],randomtemp[2]])
    usefulDDI=np.mat(usefulDDI)
    return usefulDDI,labels,DDIstructure_1,DDIstructure_2,DDIinteraction
















