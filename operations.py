
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
        type = input.type()
        print('ComP===============',self.count,input.type())
        device = input.device
        # input = input.cuda()
        if(input.type() == 'torch.FloatTensor'):
            input = input.cuda()
        else:
            scale = input.q_scale()
            zp = input.q_zero_point()
            input = input.int_repr().float()


            # ------------------to be continue------------------
            
            # --------------------------------------------------


        if(type != 'torch.FloatTensor'):
            input = (input-zp)*scale
            input=torch.quantize_per_tensor(input,scale,zp,dtype=torch.quint8)
        input = input.to(device)
        return input



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
        # print('quan')
        In_max = input.max()
        In_min = input.min()
        c = part_quant(input, In_max, In_min, self.args.actbit)
        input1 = c[0]*c[1]+c[2]

        rr = F.conv2d(input1, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        return rr




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
