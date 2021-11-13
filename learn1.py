import time
import argparse
import numpy as np
import torch
from torch import optim
from datas import getdata,gettvt
from models1 import CP
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,recall_score
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils import data
from torch.autograd import Variable
import math
#训练参数设置
parser=argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=256)
parser.add_argument('--init', default=1e-3, type=float,help="Initial scale")
parser.add_argument('--learning_rate', default=0.0001, type=float,help="Learning rate")
parser.add_argument('--decay1', default=0.9, type=float,help="decay rate for the first moment estimate in Adam")
parser.add_argument('--decay2', default=0.999, type=float,help="decay rate for second moment estimate in Adam")
args=parser.parse_args()
#数据导入
class MyDataset(data.Dataset):
    def __init__(self, data1, data2, data3, labels):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.labels = labels

    def __getitem__(self, index):
        img1, img2, img3, target = self.data1[index], self.data2[index], self.data3[index], self.labels[index]
        img1=np.array(img1)
        img1=torch.from_numpy(img1)
        img2 = np.array(img2)
        img2 = torch.from_numpy(img2)
        img3 = np.array(img3)
        img3 = torch.from_numpy(img3)
        return img1, img2, img3, target

    def __len__(self):
        return len(self.data1)
traindata,validdata,testdata=gettvt()
DDISET_tr,labels_tr,DDIstructure_1_tr,DDIstructure_2_tr,DDIinteraction_tr=getdata(traindata)
labels_tr=np.array(labels_tr)
labels_tr=np.mat(labels_tr)
labels_tr=labels_tr.transpose()
labels_tr=torch.from_numpy(labels_tr)
train_dataset=MyDataset(DDIstructure_1_tr,DDIstructure_2_tr,DDIinteraction_tr,labels_tr)
train_loader=DataLoader(dataset=train_dataset,batch_size=5000,shuffle=True)

DDISET_va,labels_va,DDIstructure_1_va,DDIstructure_2_va,DDIinteraction_va=getdata(validdata)
labels_va=np.array(labels_va)
labels_va=np.mat(labels_va)
labels_va=labels_va.transpose()
labels_va=torch.from_numpy(labels_va)
valid_dataset=MyDataset(DDIstructure_1_va,DDIstructure_2_va,DDIinteraction_va,labels_va)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=5000,shuffle=True)

DDISET_te,labels_te,DDIstructure_1_te,DDIstructure_2_te,DDIinteraction_te=getdata(testdata)
labels_te=np.array(labels_te)
labels_te=np.mat(labels_te)
labels_te=labels_te.transpose()
labels_te=torch.from_numpy(labels_te)
test_dataset=MyDataset(DDIstructure_1_te,DDIstructure_2_te,DDIinteraction_te,labels_te)
test_loader=DataLoader(dataset=test_dataset,batch_size=5000,shuffle=True)

def binary_evaluation_result(label_true, score_predict):
    roc_auc_score1 = metrics.roc_auc_score(label_true, score_predict)
    precision, recall, _ = metrics.precision_recall_curve(
        label_true, score_predict)
    pr_auc_score = metrics.auc(recall, precision)
    return pr_auc_score

#模型
model=CP(10)
#优化过程
optimizer=optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2))
#损失函数
loss = torch.nn.MSELoss(reduction='mean')
#测试
def test(DDI_test,model):
    model.eval()
    y_pred = []
    y_label = []
    for step, (data1, data2, data3, label) in enumerate(DDI_test):
        data1, data2, data3, label = (Variable(data1).float(), Variable(data2).float(), Variable(data3).float(), Variable(label).float())
        output,_=model(data1, data2, data3)
        truth = label
        truth = truth.numpy()
        y_label = y_label + truth.flatten().tolist()
        y_pred = y_pred + output.flatten().tolist()
        print("\rtest进度：%d" % (step), end=' ')
    y_pred1 = np.array(y_pred)
    y_label1 = np.array(y_label)
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label1, y_pred1.round(),average='micro'),f1_score(y_label1, y_pred1.round(),average='micro'),recall_score(y_label1, y_pred1.round(),average='micro'),binary_evaluation_result(y_label1, y_pred1.round()),accuracy_score(y_label1, y_pred1.round())




#训练
loss_history = []
t_total=time.time()
minauc=0
for epoch in range(args.epochs):
    t = time.time()
    y_pred_train = []
    y_label_train = []
    #y_pred_valid = []
    #y_label_valid = []
    for step, (data1, data2, data3, label) in enumerate(train_loader):
        model.train(True)
        data1, data2, data3, label = (Variable(data1).float(), Variable(data2).float(),Variable(data3).float(),Variable(label).float())
        #print(data1,data2,data3,label)
        #print('------------------------------------------------------')
        predictions, factors = model(data1,data2,data3)
        truth=label
        l = loss(predictions,truth)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print("\r进度：%d" %(step),end=' ')
        truth=truth.numpy()
        y_label_train = y_label_train + truth.flatten().tolist()
        y_pred_train = y_pred_train + predictions.flatten().tolist()
    roc_train = roc_auc_score(y_label_train, y_pred_train)
    print(roc_train)
    #加入测试集
    roc, precision, f1, recall, aupr, accuracy = test(valid_loader,model)
    print(roc,precision,aupr, accuracy)
    roc, precision, f1, recall, aupr, accuracy = test(test_loader,model)
    print(roc, precision, aupr, accuracy)













