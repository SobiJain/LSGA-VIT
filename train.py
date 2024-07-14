from __future__ import print_function
import cv2
import math
import numpy as np
from patchify import patchify
from sklearn.model_selection import train_test_split
from visdom import Visdom
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import model
from Utils import *
from option import opt
import time
from operator import truediv
from calflops import calculate_flops
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = False

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = opt.datasetname
X, y = loadData(dataset)
H = X.shape[0]
W = X.shape[1]
pca_components = opt.spectrumnum
print('Hyperspectral data shape:', X.shape)
print('Label shape:', y.shape)
sample_number = np.count_nonzero(y)
print('the number of sample:', sample_number)
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA :', X_pca.shape)
[nRow, nColumn, nBand] = X_pca.shape
num_class = int(np.max(y))
windowsize = opt.windowsize
Wid = opt.inputsize
halfsizeTL = int((Wid-1)/2)
halfsizeBR = int((Wid-1)/2)
paddedDatax = cv2.copyMakeBorder(X_pca, halfsizeTL, halfsizeBR, halfsizeTL, halfsizeBR, cv2.BORDER_CONSTANT, 0)  #cv2.BORDER_REPLICAT周围值
paddedDatay = cv2.copyMakeBorder(y, halfsizeTL, halfsizeBR, halfsizeTL, halfsizeBR, cv2.BORDER_CONSTANT, 0)
patchIndex = 0
X_patch = np.zeros((sample_number, Wid, Wid, pca_components))
y_patch = np.zeros(sample_number)
for h in range(0, paddedDatax.shape[0]):
    for w in range(0, paddedDatax.shape[1]):
        if paddedDatay[h, w] == 0:
            continue
        X_patch[patchIndex, :, :, :] = paddedDatax[h-halfsizeTL:h+halfsizeBR+1, w-halfsizeTL:w+halfsizeBR+1, :]
        X_patch[patchIndex] = paddedDatay[h, w]
        patchIndex = patchIndex + 1
X_train_p = patchify(paddedDatax, (Wid, Wid, pca_components), step=1)
if opt.input3D:
    X_train_p = X_train_p.reshape(-1, Wid, Wid, pca_components, 1)
else:
    X_train_p = X_train_p.reshape(-1, Wid, Wid, pca_components)
y_train_p = y.reshape(-1)
indices_0 = np.arange(y_train_p.size)
X_train_q = X_train_p[y_train_p > 0, :, :, :]
y_train_q = y_train_p[y_train_p > 0]
indices_1 = indices_0[y_train_p > 0]
y_train_q -= 1
X_train_q = X_train_q.transpose(0, 3, 1, 2)
Xtrain, Xtest, ytrain, ytest, idx1, idx2 = train_test_split(X_train_q, y_train_q, indices_1,
                                                            train_size=opt.numtrain, random_state=opt.random_seed,
                                                            stratify=y_train_q)

print(Xtest.size)
print(ytest.size)

print(Xtrain.size)
print(ytrain.size)

Xval, Xtest, yval, ytest= train_test_split(Xtest, ytest,
                                                          train_size=opt.numtrain, random_state=opt.random_seed,
                                                            stratify=ytest)                                                           

print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)
print('Xval  shape: ', Xval.shape)

BATCH_SIZE_TRAIN = opt.batchSize

# 创建train_loader和 test_loader
X = TestDS(X, y_train_q)
trainset = TrainDS(Xtrain, ytrain)
testset = TestDS(Xtest, ytest)
valset = TrainDS(Xval, yval)
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size=BATCH_SIZE_TRAIN,
                                            shuffle=True,
                                            num_workers=0,
                                            )
test_loader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=BATCH_SIZE_TRAIN,
                                            shuffle=False,
                                            num_workers=0,
                                          )
val_loader = torch.utils.data.DataLoader(dataset=valset,
                                            batch_size=BATCH_SIZE_TRAIN,
                                            shuffle=True,
                                            num_workers=0,
                                            )
all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                            batch_size=BATCH_SIZE_TRAIN,
                                            shuffle=False,
                                            num_workers=0,
                                          )

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len

def train(train_loader, val_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = model.LSGAVIT(img_size=Wid,
                         patch_size=7,
                         in_chans=pca_components,
                         num_classes=num_class,
                         embed_dim=120,
                         depths=[2],
                         num_heads=[12, 12, 12, 24],
                         ).to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))
        val_acc_list = []
        val_epoch_list = []
        val_num = val_loader.dataset.__len__()
        ## valuation
        if (epoch+1)%4 == 0 or (epoch+1)==epochs:
            val_acc =0
            net.eval()
            for batch_idx, (data, target) in enumerate(val_loader):
                data,target = data.to(device),target.to(device)
                out = net(data)
                target = target - 1  ## class 0 in out is class 1 in target
                _,pred = torch.max(out,dim=1)
                val_acc += (pred == target).sum().item()
            val_acc_list.append(val_acc/val_num)
            val_epoch_list.append(epoch)
            print(f"epoch {epoch}/{epochs}  val_acc:{val_acc_list[-1]}")
            save_name = os.path.join('/', f"epoch_{epoch}_acc_{val_acc_list[-1]:.4f}.pth")
            torch.save(net.state_dict(),save_name)

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':

    tic1 = time.perf_counter()
    net, device = train(train_loader, val_loader, epochs=200)
    # 只保存模型参数
    print("_______________________________________")
    input_shape = (128, 36, 7, 7)
    flops, macs, params = calculate_flops(model=net, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    print("Alexnet FLOPs:%s  -- MACs:%s   -- Params:%s \n" %(flops, macs, params))

    toc1 = time.perf_counter()
    
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    file_name = "data/classification_report_IP "+str(opt.numtrain)+".txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    # get_cls_map.get_cls_map(net, device, all_data_loade
