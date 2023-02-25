# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# filename = 'cnn-torch/log/SGD_lr_0.01_epoch_20.txt'
# loss, train_acc, test_acc = [], [], []
# with open(filename, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         value = [float(s) for s in line.split(':')]
#         loss.append(value[0])
#         train_acc.append(value[1])
#         test_acc.append(value[2])
# x = range(1, len(loss)+1)
# # fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=True)
# # # 设置标题
# # axs[0].set_title('Loss Function(Adam)')
# # axs[1].set_title('Training and Testing Accuracy(Adam)')
# # axs[0].set_xlabel('epoch')
# # axs[0].set_ylabel('loss')
# # axs[1].set_xlabel('epoch')
# # axs[1].set_ylabel('acc')
# # # axs[0].plt.ylim(0, 1)
# # # axs[1].plt.ylim(0, 1)
# # axs[0].plot(x, loss)
# # axs[1].plot(x, train_acc, 'r', x, test_acc, 'b')
# # plt.show()
# plt.plot(x, train_acc, 'r', x, test_acc, 'b')
# plt.savefig('cnn-torch/log/myfile_test')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
'''训练和测试精度比较'''
# filename = 'cnn-torch/log/SGD_lr_0.01_epoch_20_test.txt'
# train_acc, train_loss, val_acc, val_loss = [], [], [], []
# with open(filename, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         value = [float(s) for s in line.split(':')]
#         train_loss.append(value[0])
#         train_acc.append(value[1])
#         val_loss.append(value[2])
#         val_acc.append(value[3])
# x = range(1, len(train_acc)+1)
# # fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=True)
# # # 设置标题
# # axs[0].set_title('Loss Function(Adam)')
# # axs[1].set_title('Training and Testing Accuracy(Adam)')
# # axs[0].set_xlabel('epoch')
# # axs[0].set_ylabel('loss')
# # axs[1].set_xlabel('epoch')
# # axs[1].set_ylabel('acc')
# # # axs[0].plt.ylim(0, 1)
# # # axs[1].plt.ylim(0, 1)
# # axs[0].plot(x, loss)
# # axs[1].plot(x, train_acc, 'r', x, test_acc, 'b')
# # plt.show()
# plt.title("Training and Testing Accuracy")
# plt.xlabel("epoch")
# plt.ylabel("Accuracy")
# plt.plot(x, train_acc, 'r', label='train accuracy')
# plt.plot(x, val_acc, 'b', label='test accuracy')
# plt.legend()
# plt.savefig('cnn-torch/log/myfile_test_acc.png')

'''随机抽取图像测试'''
# import os
# import torch
# import torch.nn as nn # torchh中最重要的模块，封装了神经网络相关嗯等函数
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torchvision import transforms
# from torchvision import datasets
# from model import ConvNet
# from numpy import *
# test_file = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
# test_data = test_file.data
# test_targets = test_file.targets
# plt.figure(figsize=(9,9))
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.title(test_targets[i].numpy(), fontdict={'weight':'normal','size': 20})
#     plt.axis('off')
#     plt.imshow(test_data[i], cmap='gray')
# plt.savefig('cnn-torch/log/myfile_test.png')

'''梯度范数图'''
# filename = 'cnn-torch/log/SGD_lr_0.01_epoch_20_test_grad.txt'
# f_w, f_b, c_w, c_b = [], [], [], []
# with open(filename, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         value = [float(s) for s in line.split(':')]
#         f_w.append(value[0])
#         f_b.append(value[1])
#         c_w.append(value[2])
#         c_b.append(value[3])
# x = range(1, len(f_w)+1)
# plt.title("L2 Norm of Bias of Classification Layer")
# plt.xlabel("epoch")
# plt.ylabel("norm")
# plt.plot(x, c_w, 'r')
# # plt.plot(x, val_acc, 'b', label='test accuracy')
# # plt.legend()
# plt.savefig('cnn-torch/log/myfile_test_c_b.png')


'''比较不同学习率'''
# filename_ = 'cnn-torch/log/SGD_lr_0.1_epoch_20.txt'
# filename_1 = 'cnn-torch/log/SGD_lr_0.05_epoch_20.txt'
# filename_11 = 'cnn-torch/log/SGD_lr_0.01_epoch_20.txt'
# filename_111 = 'cnn-torch/log/SGD_lr_0.001_epoch_20.txt'
# train_acc_, val_acc_, train_acc_1, val_acc_1, train_acc_11, val_acc_11, train_acc_111, val_acc_111,= [], [], [], [], [], [], [], []
# with open(filename_, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         value = [float(s) for s in line.split(':')]
#         train_acc_.append(value[1])
#         val_acc_.append(value[2])
# with open(filename_1, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         value = [float(s) for s in line.split(':')]
#         train_acc_1.append(value[1])
#         val_acc_1.append(value[2])
# with open(filename_11, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         value = [float(s) for s in line.split(':')]
#         train_acc_11.append(value[1])
#         val_acc_11.append(value[2])
# with open(filename_111, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         value = [float(s) for s in line.split(':')]
#         train_acc_111.append(value[1])
#         val_acc_111.append(value[2])
# x = range(1, len(train_acc_)+1)


# plt.title("Train Accuracy of Different Learning Rate")
# plt.xlabel("epoch")
# plt.ylabel("Train Accuracy")
# plt.plot(x, train_acc_, 'r', label='rl=0.1')
# plt.plot(x, train_acc_1, 'b', label='rl=0.05')
# plt.plot(x, train_acc_11, 'y', label='rl=0.01')
# plt.plot(x, train_acc_111, 'g', label='rl=0.001')
# plt.legend()
# plt.savefig('cnn-torch/log/myfile_rl_train.png')

'''比较不同优化器'''
filename_sgd = 'cnn-torch/log/SGD_lr_0.01_epoch_20.txt'
filename_rms = 'cnn-torch/log/RMS_lr_0.01_epoch_20.txt'
filename_adam = 'cnn-torch/log/Adam_lr_0.01_epoch_20.txt'
train_acc_sgd, val_acc_sgd, train_acc_rms, val_acc_rms, train_acc_adam, val_acc_adam = [], [], [], [], [], []
with open(filename_sgd, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split(':')]
        train_acc_sgd.append(value[1])
        val_acc_sgd.append(value[2])
with open(filename_rms, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split(':')]
        train_acc_rms.append(value[1])
        val_acc_rms.append(value[2])
with open(filename_adam, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split(':')]
        train_acc_adam.append(value[1])
        val_acc_adam.append(value[2])

x = range(1, len(train_acc_sgd)+1)


plt.title("Test Accuracy of Different Optimizer")
plt.xlabel("epoch")
plt.ylabel("Test Accuracy")
plt.plot(x, val_acc_sgd, 'r', label='SGD')
plt.plot(x, val_acc_rms, 'b', label='RMSprop')
plt.plot(x, val_acc_adam, 'g', label='Adam')
plt.legend()
plt.savefig('cnn-torch/log/myfile_opt_test.png')