import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from Models import shufflenetv2 as shufflenetv2
import sys
from .utils import _replace_relu, quantize_model
from operations import ComP

# shufflenetv2 = sys.modules['torchvision.models.shufflenetv2']

__all__ = [
    'QuantizableShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

quant_model_urls = {
    'shufflenetv2_x0.5_fbgemm': None,
    'shufflenetv2_x1.0_fbgemm':
        'https://download.pytorch.org/models/quantized/shufflenetv2_x1_fbgemm-db332c57.pth',
    'shufflenetv2_x1.5_fbgemm': None,
    'shufflenetv2_x2.0_fbgemm': None,
}


class QuantizableInvertedResidual(shufflenetv2.InvertedResidual):
    def __init__(self, opts, *args, **kwargs):
        super(QuantizableInvertedResidual, self).__init__(opts,*args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()
        self.comp1 = nn.Sequential()
        self.comp2 = nn.Sequential()
        if self.stride == 1:
            num2 = 0
            for i in range(len(self.branch2)):
                if not (str(self.branch2[i])[0:8] == 'Identity'):
                    self.comp2.add_module('comp{}'.format(num2),ComP(opts))
                    num2+=1
        else:
            num1 = 0
            for i in range(len(self.branch1)):
                if not (str(self.branch1[i])[0:8] == 'Identity'):
                    self.comp1.add_module('comp{}'.format(num1),ComP(opts))
                    num1+=1
            num2 = 0
            for i in range(len(self.branch2)):
                if not (str(self.branch2[i])[0:8] == 'Identity'):
                    self.comp2.add_module('comp{}'.format(num2),ComP(opts))
                    num2+=1
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            num2 = 0
            for i in range(len(self.branch2)):
                if not (str(self.branch2[i])[0:8] == 'Identity'):
                    x2 = self.comp2[num2](x2)
                    num2 += 1
                x2 = self.branch2[i](x2)
            branch2 = x2
            # branch2 = self.branch2(x2)
            out = self.cat.cat((x1, branch2), dim=1)
        else:
            x1 = x
            num1 = 0
            for i in range(len(self.branch1)):
                if not (str(self.branch2[i])[0:8] == 'Identity'):
                    x1 = self.comp1[num1](x1)
                    num1+=1
                x1 = self.branch1[i](x1)
            x2 = x
            num2 = 0
            for i in range(len(self.branch2)):
                if not (str(self.branch2[i])[0:8] == 'Identity'):
                    x2 = self.comp2[num2](x2)
                x2 = self.branch2[i](x2)
                num2+=1
            branch2 = x2
            branch1 = x1
            # branch1 = self.branch1(x)
            # branch2 = self.branch2(x)
            out = self.cat.cat((branch1, branch2), dim=1)

        out = shufflenetv2.channel_shuffle(out, 2)

        return out


class QuantizableShuffleNetV2(shufflenetv2.ShuffleNetV2):
    def __init__(self, opts, *args, **kwargs):
        super(QuantizableShuffleNetV2, self).__init__(opts, *args, inverted_residual=QuantizableInvertedResidual, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in shufflenetv2 model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for name, m in self._modules.items():
            if name in ["conv1", "conv5"]:
                torch.quantization.fuse_modules(m, [["0", "1", "2"]], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableInvertedResidual:
                if len(m.branch1._modules.items()) > 0:
                    torch.quantization.fuse_modules(
                        m.branch1, [["0", "1"], ["2", "3", "4"]], inplace=True
                    )
                torch.quantization.fuse_modules(
                    m.branch2,
                    [["0", "1", "2"], ["3", "4"], ["5", "6", "7"]],
                    inplace=True,
                )
    def loadPreTrained(self):
        self.load_state_dict(torch.load('qmodels/shufflenetv2_x1_fbgemm-db332c57.pth'))


def _shufflenetv2(opts, arch, pretrained, progress, quantize, *args, **kwargs):
    model = QuantizableShuffleNetV2(opts, *args, **kwargs)
    _replace_relu(model)

    if quantize:
        # TODO use pretrained as a string to specify the backend
        backend = 'fbgemm'
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    # if pretrained:
    #     if quantize:
    #         model_url = quant_model_urls[arch + '_' + backend]
    #     else:
    #         model_url = shufflenetv2.model_urls[arch]

    #     state_dict = load_state_dict_from_url(model_url,
    #                                           progress=progress)

    #     model.load_state_dict(state_dict)
    return model


def shufflenet_v2_x0_5(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress, quantize,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(opts, pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _shufflenetv2(opts, 'shufflenetv2_x1.0', pretrained, progress, quantize,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress, quantize,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress, quantize,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
