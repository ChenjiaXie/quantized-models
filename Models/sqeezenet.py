from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init

# from ..utils import _log_api_usage_once

from operations import ReLUX

__all__ = ["SqueezeNet", "squeezenet1_0", "squeezenet1_1"]

model_urls = {
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
}


from torch.nn import functional as F
class ConvX2d(nn.Conv2d):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    layer = 0
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
        super(ConvX2d, self).__init__(in_channels, out_channels, kernel_size, stride,padding, dilation, groups, bias)
        self.layer = ConvX2d.layer
        ConvX2d.layer += 1

    def forward(self, x):
        output = F.conv2d(x, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        return output


class Fire(nn.Module):
    def __init__(self, args,inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = ConvX2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = ReLUX(args, inplace=True)
        self.expand1x1 = ConvX2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = ReLUX(args, inplace=True)
        self.expand3x3 = ConvX2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = ReLUX(args, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, args, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                ConvX2d(3, 96, kernel_size=7, stride=2),
                ReLUX(args, inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(args,96, 16, 64, 64),
                Fire(args,128, 16, 64, 64),
                Fire(args,128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(args,256, 32, 128, 128),
                Fire(args,256, 48, 192, 192),
                Fire(args,384, 48, 192, 192),
                Fire(args,384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(args,512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                ConvX2d(3, 64, kernel_size=3, stride=2),
                ReLUX(args, inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = ConvX2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, ReLUX(args, inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, ConvX2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(args,version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(args,version, **kwargs)
    if pretrained:
        arch = "squeezenet" + version
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url(model_urls[arch]))
    return model


def squeezenet1_0(args,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    The required minimum input size of the model is 21x21.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet(args,"1_0", pretrained, progress, **kwargs)


def squeezenet1_1(args,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    The required minimum input size of the model is 17x17.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet(args,"1_1", pretrained, progress, **kwargs)
