import os
import torch
import torch.nn as nn # torch中最重要的模块，封装了神经网络相关嗯等函数
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import datasets
from model import ConvNet
from numpy import *

# 设置超参数
BATCH_SIZE = 64 # 每批训练的样本数量
EPOCHS = 20 # 总共训练迭代的次数
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
learning_rate = 0.005 # 设定初始的学习率
MODEL_DIRECTORY = 'cnn-torch/model'
DATA_DIRECTOTY = 'cnn-torch/log'
MODEL_EXIST = False

# 加载训练集
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, ), std=(0.5, ))])), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, ), std=(0.5, ))])), batch_size=BATCH_SIZE, shuffle=True)

# 训练前准备
ConvModel = ConvNet().to(DEVICE) # 初始化模型，将网络操作移动到GPU或CPU
if MODEL_EXIST:
    with open(MODEL_DIRECTORY, 'rb') as anno_file:
        ConvModel.load_state_dict(torch.load(anno_file))

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.RMSprop(ConvModel.parameters(), lr=learning_rate, alpha=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1) #定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子


# 训练函数
def train(num_epochs, _model, _device, _train_loader, _optimizer, _lr_scheduler):
    _model.train() # 设置模型为训练模式
    Loss = []
    train_acc = []
    test_acc = []
    test_loss = []
    
    # 存储梯度范数
    feature_w1 = []
    feature_b1 = []
    class_w1 = []
    class_b1 = []
    for epoch in range(num_epochs):
        Loss1 = 0.0
        correct = 0.0
        
        feature_w = 0.0
        feature_b = 0.0
        class_w = 0.0
        class_b = 0.0
        for i, (images, labels) in enumerate(_train_loader):
            samples = images.to(_device)
            labels = labels.to(_device)
            output = _model(samples.reshape(-1, 1, 28, 28))
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()  # .cpu()是将参数迁移到cpu上来。

            loss = criterion(output, labels)
            Loss1 += loss.item()
            optimizer.zero_grad() # 优化器内部参数梯度必须变为0
            loss.backward() # 损失值后向传播
            '''存储参数梯度大小'''
        #     # if (i + 1) % 100 == 0:
        #     #     for name, parms in _model.named_parameters():
        #     #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #     #               ' -->grad_value:', torch.norm(parms.grad).item())
        #     #         print(parms.grad.shape) 
        #     for name, parms in _model.named_parameters():
        #         if name=='features.0.weight':
        #             feature_w += torch.norm(parms.grad).item()
        #         if name=='features.0.bias':
        #             feature_b += torch.norm(parms.grad).item()
        #         if name=='classifier.1.weight':
        #             class_w += torch.norm(parms.grad).item()
        #         if name=='classifier.1.bias':
        #             class_b += torch.norm(parms.grad).item()      
            optimizer.step()
            if (i + 1) % 100 == 0:
                # Loss1.append(loss.data.cpu().numpy())
                print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch + 1, num_epochs, i + 1, loss.item()))
        # feature_w1.append(feature_w / len(_train_loader.dataset))
        # feature_b1.append(feature_b / len(_train_loader.dataset))
        # class_w1.append(class_w / len(_train_loader.dataset))
        # class_b1.append(class_b / len(_train_loader.dataset))
        
        Loss.append(Loss1 / len(_train_loader.dataset))
        train_acc.append(100. * correct / len(_train_loader.dataset))
        t_loss, t_acc = test(test_loader, _model, DEVICE)
        test_loss.append(t_loss)
        test_acc.append(t_acc)
        # test_acc.append(test(test_loader, _model, DEVICE))
        # _lr_scheduler.step()  # 设置学习调度器开始准备更新
    # Loss0 = torch.tensor(Loss)
    save_path = os.path.join(DATA_DIRECTOTY, "RMS_lr_{}_epoch_{}.txt".format(learning_rate, num_epochs))
    with open(save_path, "a") as f:
        for i in range(len(train_acc)):
            f.write('{:}:{:}:{:}\n'.format(Loss[i], train_acc[i], test_acc[i]))
            #f.write('{:}:{:}:{:}:{:}\n'.format(Loss[i], train_acc[i], test_loss[i],test_acc[i]))
            #f.write('{:}:{:}:{:}:{:}\n'.format(feature_w1[i], feature_b1[i], class_w1[i], class_b1[i]))
        f.close()


    # torch.save(Loss0, 'log/epoch_{}'.format(epoch + 1))
    # 保存网络
    torch.save(_model.state_dict(), os.path.join(MODEL_DIRECTORY, 'net_params_' + 'epoch_{}'.format(epoch+1) + '.pth'))

# 测试函数
def test(_test_loader, _model, _device):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0

    with torch.no_grad():  # 如果不需要 backward更新梯度，那么就要禁用梯度计算，减少内存和计算资源浪费。
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = ConvModel(data.reshape(-1, 1, 28, 28))
            loss += criterion(output, target).item()  # 添加损失值
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # .cpu()是将参数迁移到cpu上来。

    loss /= len(_test_loader.dataset)

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))
    return (loss, 100. * correct / len(_test_loader.dataset))

if not MODEL_EXIST:
    train(EPOCHS, ConvModel, DEVICE, train_loader, optimizer, exp_lr_scheduler)
    test(test_loader, ConvModel, DEVICE)
    # test(train_loader, ConvModel, DEVICE)
else:
    test(test_loader, ConvModel, DEVICE)




