import math
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn
from torchvision.models.inception import (
    BasicConv2d,
    InceptionA,
    InceptionAux,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
)

inception_urls = {
    "Inception_v3": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
}


class InceptionDesigner(nn.Module):
    def __init__(
        self,
        input_size,
        InceptionBlocks: Optional[List[Callable[..., nn.Module]]] = None,
        num_classes=1000,
        aux_logits: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super(InceptionDesigner, self).__init__()

        if InceptionBlocks is None:
            InceptionBlocks = [
                BasicConv2d,
                InceptionA,
                InceptionB,
                InceptionC,
                InceptionD,
                InceptionE,
                InceptionAux,
            ]

        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        input_tensor = self._transform_input(input_size)
        out = self.Conv2d_1a_3x3(input_tensor)
        out = self.Conv2d_2a_3x3(out)
        out = self.Conv2d_2b_3x3(out)
        out = self.maxpool1(out)
        out = self.Conv2d_3b_1x1(out)
        out = self.Conv2d_4a_3x3(out)
        out = self.maxpool2(out)
        out = self.Mixed_5b(out)
        out = self.Mixed_5c(out)
        out = self.Mixed_6a(out)
        out = self.Mixed_6b(out)
        out = self.Mixed_6c(out)
        out = self.Mixed_6d(out)
        out = self.Mixed_6e(out)
        if aux_logits:
            out = self.AuxLogits
        out = self.Mixed_7a(out)
        out = self.Mixed_7b(out)
        out = self.Mixed_7c(out)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x
