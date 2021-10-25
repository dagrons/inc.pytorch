import torch # Tensor, Variable
import torch.nn as nn
from torchinfo import summary

"""
## 参数解释
1. nn.Conv2d(in_channels, out_channels, kernel_size=<odd>, stride, padding=<kernel_size/2>)
(in_channels, H, W) => (out_channels, H/stride, W/stride) 为了简化计算，通常选H和W为stride的倍数（在不考虑> > > dilation的情况下），padding一般选择lower_bound(kernel_size/2)
2. BatchNorm2d是为了让模型更快收敛
3. MaxPool2d和Conv2d区别在于，MaxPool2d的filter已经确定，而不是需要学习的参数
"""

class BasicBlock(nn.Module):
    # 原图像变为1/stride
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        )
        
        self.shortcut = nn.Sequential()
        
        if stride!=1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.shortcut(x) + self.residual_function(x))


class IncResNet(nn.Module):
    # (3, 64, 64) => (512) => (9)
    def __init__(self, num_block, base_num_classes):
        super().__init__()        
        
        self.in_channels = 64

        self.n_cls = base_num_classes
        
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 将RGB三通道转化为64channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ) # (3, H, W) => (64, H/4, W/4)
        # 假设原图像为3通道
        
        self.conv_block1 = self.generate_layer(64, num_block[0], 2) # (in_channels, H, W) => (64, H, W)
        self.conv_block2 = self.generate_layer(128, num_block[1], 2) # (in_channels, H, W) => (128, H/2, W/2)
        self.conv_block3 = self.generate_layer(256, num_block[2], 2) # (in_channels, H, W) => (256, H/2, W/2)
        self.conv_block4 = self.generate_layer(512, num_block[3], 2) # (in_channels, H, W) => (512, H/2, W/2)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3) # (H, W) => (H, W)
        
        self.fc = nn.Linear(512, base_num_classes) # 512 => 9 
        # 假设了原图像为32x32
        
         # Kaiming He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def add_classes(self, num_new_cls):
        with torch.no_grad():            
            old_fc_weights = self.fc.weight
            old_fc_bias = self.fc.bias

            self.fc = nn.Linear(512, self.n_cls + num_new_cls)
            self.fc.weight[:self.n_cls] = old_fc_weights
            self.fc.bias[:self.n_cls] = old_fc_bias
            self.n_cls += num_new_cls    
        
    def generate_layer(self, out_channels, num_block, stride):
        # 生成重复的blocks，除了第一个block，后面block保持图像大小不变，第一个block将图像缩小为1/stride
        # 即第一个block会下采样，而后面的只是特征变换
        # 总体来说，(in_channels, H, W) => (out_channels, 1/stride, 1/stride)
        """
        out_channels: output channels of this layer
        num_block: the number of blocks per layer
        stride: stride of first layer can be 1 or 2, the others would always be 1
        """
        
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv_input(x)
        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.conv_block3(output)
        output = self.conv_block4(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1) # padding         
        output = self.fc(output)
        
        return output


if __name__ == "__main__":
    model = IncResNet(num_block=[3, 4, 6, 3], base_num_classes=20)
    #print (model)    
    for name, module in model.named_children():
        print (name)
    #summary(model, input_size=(10, 3, 64, 64))