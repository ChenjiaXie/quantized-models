from torch import nn
from torch import Tensor
import torch

# from ..._internally_replaced_utils import load_state_dict_from_url

from typing import Any

from .mobilenetv2 import InvertedResidual, MobileNetV2, model_urls
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from .utils import _replace_relu, quantize_model
from .misc import ConvNormActivation
from operations import ComP


__all__ = ['QuantizableMobileNetV2', 'mobilenet_v2']

quant_model_urls = {
    'mobilenet_v2_qnnpack':
        'https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth'
}


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self,opts, *args: Any, **kwargs: Any) -> None:
        super(QuantizableInvertedResidual, self).__init__(opts,*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()
        self.comp = nn.Sequential()
        for i in range(len(self.conv)):
            if not(str(self.conv[i])[0:8] == 'Identity'):
                self.comp.add_module('Comp{}'.format(i),ComP(opts))

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            convx = x
            num = 0
            for i in range(len(self.conv)):
                if not(str(self.conv[i])[0:8] == 'Identity'):
                    convx = self.comp[num](convx)
                    num += 1
                convx = self.conv[i](convx)
            return self.skip_add.add(x, convx)
        else:
            convx = x
            num = 0
            for i in range(len(self.conv)):
                if not(str(self.conv[i])[0:8] == 'Identity'):
                    convx = self.comp[num](convx)
                    num += 1
                convx = self.conv[i](convx)
            return convx

    def fuse_model(self) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)


class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, opts, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super(QuantizableMobileNetV2, self).__init__(opts,*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        for m in self.modules():
            if type(m) == ConvNormActivation:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QuantizableInvertedResidual:
                m.fuse_model()
    def loadPreTrained(self):
        self.load_state_dict(torch.load('qmodels/mobilenet_v2_qnnpack_37f702c5.pth'))


def mobilenet_v2(
    opts,
    pretrained: bool = False,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_.

    Note that quantize = True returns a quantized model with 8 bit
    weights. Quantized models only support inference and run on CPUs.
    GPU inference is not yet supported

    Args:
     pretrained (bool): If True, returns a model pre-trained on ImageNet.
     progress (bool): If True, displays a progress bar of the download to stderr
     quantize(bool): If True, returns a quantized model, else returns a float model
    """
    model = QuantizableMobileNetV2(opts,block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)

    if quantize:
        # TODO use pretrained as a string to specify the backend
        backend = 'qnnpack'
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    if pretrained:
        if quantize:
            model_url = quant_model_urls['mobilenet_v2_' + backend]
        else:
            model_url = model_urls['mobilenet_v2']

        # state_dict = load_state_dict_from_url(model_url,
        #                                       progress=progress)

        # model.load_state_dict(state_dict)
    return model
