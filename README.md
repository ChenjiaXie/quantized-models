# quantized-network

download .pth files to qmodels/:

googlenet           : https://download.pytorch.org/models/quantized/googlenet_fbgemm-c00238cf.pth

inception_v3        : https://download.pytorch.org/models/quantized/inception_v3_google_fbgemm-71447a44.pth

mobilenet_v2        : https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth

mobilenet_v3_large  : https://download.pytorch.org/models/quantized/mobilenet_v3_large_qnnpack-5bcacf28.pth

resnet18            : https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth

resnet50            : https://download.pytorch.org/models/quantized/resnet50_fbgemm_bf931d71.pth

resnext101          : https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm_09835ccf.pth

shufflenetv2_x1.0   : https://download.pytorch.org/models/quantized/shufflenetv2_x1_fbgemm-db332c57.pth

ghostnet            : https://1drv.ms/u/s!Ahqo_6nBJPIHhloNb-Rg2uXs38MU?e=Smakww

## To do some operations before convolution layer in these networks：

1. Ghostnet can do some operations on the feature maps of the  inter-layer  by manipulating the Class ConvX in the operations.py

2. And other works  shoud modifies the Class Comp in operations.py

# Run instructions

python validate.py --data <imagenet foder location> --model <modelname> --actbit <8 or 16 for ghostnet>


    # python validate.py --model mobilenet_v3_large --data ./imagenet

    # python validate.py --model resnet_18 --data ./imagenet

    # python validate.py --model resnet_50 --data ./imagenet

    # python validate.py --model resnext_101 --data ./imagenet

    # python validate.py --model googlenet --data ./imagenet

    # python validate.py --model shufflenet_v2 --data ./imagenet

    # python validate.py --model inception_v3 --data ./imagenet

    # python validate.py --model mobilenet_v2 --data ./imagenet

    # python validate.py --model ghostnet --data ./imagenet --actbit 8
