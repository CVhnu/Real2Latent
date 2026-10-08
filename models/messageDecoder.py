import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class RefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefinementBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.se = SEBlock(out_channels)
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out
class MessageDecoder(nn.Module):
    def __init__(self, message_len=48, pretrained_weights=None, hidden_dim=256):
        super(MessageDecoder, self).__init__()
        if pretrained_weights is not None:
            print(f"Loading backbone weights from {pretrained_weights}")
            backbone = models.resnet50(pretrained=False)
            checkpoint = torch.load(pretrained_weights)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            checkpoint = {k: v for k, v in checkpoint.items() if 'fc' not in k}
            backbone.load_state_dict(checkpoint, strict=False)
        else:
            backbone = models.resnet50(pretrained=True)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.layer1_dim = 512
        self.lat_layer1 = RefinementBlock(256, self.layer1_dim)
        self.lat_layer2 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        self.lat_layer4 = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        total_concat_dim = self.layer1_dim + hidden_dim * 3
        self.head = nn.Sequential(
            nn.Linear(total_concat_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, message_len)
        )
    def forward(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p2 = self.lat_layer1(c2)
        p3 = self.lat_layer2(c3)
        p4 = self.lat_layer3(c4)
        p5 = self.lat_layer4(c5)
        out2 = F.adaptive_avg_pool2d(p2, (1, 1)).flatten(1)
        out3 = F.adaptive_avg_pool2d(p3, (1, 1)).flatten(1)
        out4 = F.adaptive_avg_pool2d(p4, (1, 1)).flatten(1)
        out5 = F.adaptive_avg_pool2d(p5, (1, 1)).flatten(1)
        features = torch.cat([out2, out3, out4, out5], dim=1)
        out = self.head(features)
        return out



