import torch.nn.functional as F
from torch import nn


def init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class WideBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(WideBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    padding=0,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        o1 = F.relu(self.bn1(x))
        y = self.conv1(o1)
        o2 = F.relu(self.bn2(y))
        z = self.conv2(o2)
        z += self.shortcut(o1)
        return z


class WideResNet(nn.Module):
    def __init__(self, depth, width, num_classes):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "WideResNet model's depth should be 6n+4."
        n = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]
        in_planes = 16
        self.conv1 = conv3x3(3, in_planes)
        self.layer1 = self._wide_layer(in_planes, widths[0], n, stride=1)
        self.layer2 = self._wide_layer(widths[0], widths[1], n, stride=2)
        self.layer3 = self._wide_layer(widths[1], widths[2], n, stride=2)
        self.bn = nn.BatchNorm2d(widths[2])
        self.linear = nn.Linear(widths[2], num_classes)

    def _wide_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(WideBlock(in_planes, planes, stride))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x))
        x = F.avg_pool2d(x, 8, 1, 0)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
