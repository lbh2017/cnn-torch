import torch.nn as nn

# 设计模型
class ConvNet(nn.Module):
    def __init__(self): # self代表着实例本身，不能省略
        super(ConvNet, self).__init__() # 子类将父类的__init()放到自己的init()中
        # 提取特征层
        self.features = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        #分类层
        self.classifier = nn.Sequential(
            # Dropout层
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

    # 前向传递函数
    def forward(self, x):
        x = self.features(x)
        # 输出结果展平成一维向量
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
