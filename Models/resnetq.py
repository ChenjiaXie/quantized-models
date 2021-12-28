import torch
# import torchvision
# from .ResNet import Bottleneck, BasicBlock, ResNetImagenet, model_urls
# from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet, model_urls
# from .resnet1 import Bottleneck, BasicBlock, ResNet, model_urls
from .resnet import Bottleneck, BasicBlock, ResNet, model_urls
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
# from torch.quantization import QuantStub, DeQuantStub
# from .quantization import fuse_modules
from torch._jit_internal import Optional
from .utils import _replace_relu, quantize_model
from operations import ComP

__all__ = ['QuantizableResNet', 'resnet18', 'resnet50',
           'resnext101_32x8d']


quant_model_urls = {
    'resnet18_fbgemm':
        'https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth',
    'resnet50_fbgemm':
        'https://download.pytorch.org/models/quantized/resnet50_fbgemm_bf931d71.pth',
    'resnext101_32x8d_fbgemm':
        'https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm_09835ccf.pth',
}


class QuantizableBasicBlock(BasicBlock):
    def __init__(self, opts, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(opts,*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()
        self.comp1 = ComP(opts)
        self.comp2 = ComP(opts)

    def forward(self, x):
        identity = x

        # print('quan1:',x.shape)
        x = self.comp1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.comp2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableBottleneck(Bottleneck):
    def __init__(self,opts, *args, **kwargs):
        super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.comp1 = ComP(opts)
        self.comp2 = ComP(opts)
        self.comp3 = ComP(opts)

    def forward(self, x):
        identity = x

        print('quan2:',x.shape)
        
        x = self.comp1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.comp2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.comp3(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            fuse_modules(self.downsample, ['0', '1'], inplace=True)


# class QuantizableResNet(ResNetImagenet):

#     def __init__(self,opts, *args, **kwargs):
#         super(QuantizableResNet, self).__init__(opts,*args, **kwargs)
class QuantizableResNet(ResNet):

    def __init__(self,arch,opts, *args, **kwargs):
        super(QuantizableResNet, self).__init__(opts,*args, **kwargs)

# class QuantizableResNet(ResNet):

#     def __init__(self,opts, *args, **kwargs):
#         super(QuantizableResNet, self).__init__(*args, **kwargs)


        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.arch = arch

    def forward(self, x):
        x = self.quant(x)
        # print('first type',x.type())
        # Ensure scriptability
        # super(QuantizableResNet,self).forward(x)
        # is not scriptable
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBottleneck or type(m) == QuantizableBasicBlock:
                m.fuse_model()
    def loadPreTrained(self):
        if(self.arch == 'resnet18'):
            self.load_state_dict(torch.load('qmodels/resnet18_fbgemm_16fa66dd.pth'))
        if(self.arch == 'resnet50'):
            self.load_state_dict(torch.load('qmodels/resnet50_fbgemm_bf931d71.pth'))
        if(self.arch == 'resnext101_32x8d'):
            self.load_state_dict(torch.load('qmodels/resnext101_32x8_fbgemm_09835ccfcd.pth'))


# def _resnet(opts,arch, block, layers, pretrained, progress, quantize, **kwargs):
#     model = QuantizableResNet(opts,block, layers, **kwargs)
#     _replace_relu(model)
#     if quantize:
#         # TODO use pretrained as a string to specify the backend
#         backend = 'fbgemm'
#         quantize_model(model, backend)
#     else:
#         assert pretrained in [True, False]

#     # if pretrained:
#     #     if quantize:
#     #         model_url = quant_model_urls[arch + '_' + backend]
#     #     else:
#     #         model_url = model_urls[arch]

#     #     state_dict = load_state_dict_from_url(model_url,
#     #                                           progress=progress)

#     #     model.load_state_dict(state_dict)
#     return model


# def resnet18(opts,pretrained=True, progress=True, quantize=True, **kwargs):
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#         quantize (bool): If True, return a quantized version of the model
#     """
#     return _resnet(opts,'resnet18', QuantizableBasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    quantize, **kwargs)


# def resnet50(pretrained=False, progress=True, quantize=False, **kwargs):
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#         quantize (bool): If True, return a quantized version of the model
#     """
#     return _resnet('resnet50', QuantizableBottleneck, [3, 4, 6, 3], pretrained, progress,
#                    quantize, **kwargs)


# def resnext101_32x8d(pretrained=False, progress=True, quantize=False, **kwargs):
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#         quantize (bool): If True, return a quantized version of the model
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', QuantizableBottleneck, [3, 4, 23, 3],
#                    pretrained, progress, quantize, **kwargs)

def _resnet(opts,arch, block, layers, pretrained, progress, quantize, **kwargs):
    model = QuantizableResNet(opts,arch,block, layers, **kwargs)
    _replace_relu(model)
    if quantize:
        # TODO use pretrained as a string to specify the backend
        backend = 'fbgemm'
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    if pretrained:
        if quantize:
            model_url = quant_model_urls[arch + '_' + backend]
        else:
            model_url = model_urls[arch]

        # state_dict = load_state_dict_from_url(model_url,
        #                                       progress=progress)

        # model.load_state_dict(state_dict)
    return model


def resnet18(opts,pretrained=True, progress=True, quantize=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _resnet(opts,'resnet18', QuantizableBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   quantize, **kwargs)


def resnet50(opts,pretrained=True, progress=True, quantize=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _resnet(opts,'resnet50', QuantizableBottleneck, [3, 4, 6, 3], pretrained, progress,
                   quantize, **kwargs)


def resnext101_32x8d(opts,pretrained=False, progress=True, quantize=False, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(opts,'resnext101_32x8d', QuantizableBottleneck, [3, 4, 23, 3],
                   pretrained, progress, quantize, **kwargs)
