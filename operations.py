
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class ComP(nn.Module):
    count=0
    def __init__(self, args):
    # def __init__(self):
        super(ComP, self).__init__()
        self.args = args
        self.count = ComP.count
        ComP.count += 1

    def forward(self,input):

        return input


class ACT(nn.Module):
    layer=0
    def __init__(self):
        super(ACT, self).__init__()
        self.layer = ACT.layer
        ACT.layer += 1
        self.count = 0

class ConvX(nn.Conv2d):
    count = 0
    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ConvX, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

        self.count = ConvX.count
        self.args = args
        ConvX.count += 1

    def forward(self, input):
        input = input.cuda()
        import os 
        if not os.path.exists(f'{self.args.model}_ConvX'):
            os.mkdir(f'{self.args.model}_ConvX')
        np.save(f'{self.args.model}_ConvX/Layer{self.layer}_X.npy',np.array(input.cpu()))
        qunat_input, scale, min = part_quant(input, torch.max(input), torch.min(input), 8)
        np.save(f'{self.args.model}_ConvX/Layer{self.layer}_QX.npy',np.array(qunat_input.cpu()))
        rx = qunat_input * scale + min

        rr = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        return rr

class ReLUX(ACT):
    def __init__(self, args, mxRelu6=False, inplace = True):
        super(ReLUX, self).__init__()
        self.args = args
        if mxRelu6:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, input,inplace=True):
        self.count += 1
        input = self.relu(input)
        import os 
        if not os.path.exists(f'{self.args.model}_ReLUX'):
            os.mkdir(f'{self.args.model}_ReLUX')
        np.save(f'{self.args.model}_ReLUX/Layer{self.layer}_X.npy',np.array(input.cpu()))
        qunat_input, scale, min = part_quant(input, torch.max(input), torch.min(input), 8)
        np.save(f'{self.args.model}_ReLUX/Layer{self.layer}_QX.npy',np.array(qunat_input.cpu()))
        rx = qunat_input * scale + min

        return input

class ReLU6X(ACT):
    def __init__(self, args):
        super(ReLU6X, self).__init__()

        self.relu = nn.ReLU6(inplace=True)
        self.args = args


    def forward(self, input,inplace=True):
        input = self.relu(input)
        import os 
        if not os.path.exists(f'{self.args.model}_ReLU6X'):
            os.mkdir(f'{self.args.model}_ReLU6X')
        np.save(f'{self.args.model}_ReLU6X/Layer{self.layer}_X.npy',np.array(input.cpu()))
        qunat_input, scale, min = part_quant(input, torch.max(input), torch.min(input), 8)
        np.save(f'{self.args.model}_ReLU6X/Layer{self.layer}_QX.npy',np.array(qunat_input.cpu()))
        rx = qunat_input * scale + min
        return input

class HardX(ACT):
    def __init__(self, args):
        super(HardX, self).__init__()

        self.relu = nn.Hardswish(inplace=True)
        self.args = args

    def forward(self, input,inplace=True):
        input = self.relu(input)
        import os 
        if not os.path.exists(f'{self.args.model}_HardX'):
            os.mkdir(f'{self.args.model}_HardX')
        np.save(f'{self.args.model}_HardX/Layer{self.layer}_X.npy',np.array(input.cpu()))
        qunat_input, scale, min = part_quant(input, torch.max(input), torch.min(input), 8)
        np.save(f'{self.args.model}_HardX/Layer{self.layer}_QX.npy',np.array(qunat_input.cpu()))
        rx = qunat_input * scale + min
        return input

class SiLUX(ACT):
    def __init__(self, args):
        super(SiLUX, self).__init__()

        self.silu = nn.SiLU(inplace=True)
        self.args = args
    def forward(self, input,inplace=True):
        input = self.silu(input)
        import os 
        if not os.path.exists(f'{self.args.model}_SiLUX'):
            os.mkdir(f'{self.args.model}_SiLUX')
        np.save(f'{self.args.model}_SiLUX/Layer{self.layer}_X.npy',np.array(input.cpu()))
        qunat_input, scale, min = part_quant(input, torch.max(input), torch.min(input), 8)
        np.save(f'{self.args.model}_SiLUX/Layer{self.layer}_QX.npy',np.array(qunat_input.cpu()))
        rx = qunat_input * scale + min
        return input

class GELUX(ACT):
    def __init__(self):
        super(GELUX, self).__init__()

        self.gelu = nn.GELU()
        # self.args = args

    def forward(self, input,inplace=True):
        input = self.gelu(input)
        print(input.shape)
        import os 
        if not os.path.exists(f'{self.args.model}_GELUX'):
            os.mkdir(f'{self.args.model}_GELUX')
        np.save(f'{self.args.model}_GELUX/Layer{self.layer}_X.npy',np.array(input.cpu()))
        qunat_input, scale, min = part_quant(input, torch.max(input), torch.min(input), 8)
        np.save(f'{self.args.model}_GELUX/Layer{self.layer}_QX.npy',np.array(qunat_input.cpu()))
        rx = qunat_input * scale + min
        return input

class SigmoidX(ACT):
    def __init__(self, args):
        super(SigmoidX, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.args = args

    def forward(self, input):
        input = self.sigmoid(input)
        import os 
        if not os.path.exists(f'{self.args.model}_SigmoidX'):
            os.mkdir(f'{self.args.model}_SigmoidX')
        np.save(f'{self.args.model}_SigmoidX/Layer{self.layer}_X.npy',np.array(input.cpu()))
        qunat_input, scale, min = part_quant(input, torch.max(input), torch.min(input), 8)
        np.save(f'{self.args.model}_SigmoidX/Layer{self.layer}_QX.npy',np.array(qunat_input.cpu()))
        rx = qunat_input * scale + min
        return input



def part_quant(x, max, min, bitwidth):
    if max != min:
        Scale = (2 ** bitwidth - 1) / (max - min)
        Q_x = Round.apply((x - min) * Scale)
        return Q_x, 1 / Scale, min
    else:
        Q_x = x
        return Q_x, 1, 0

class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        round = x.round()
        return round.to(x.device)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input, None, None
