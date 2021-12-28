# rm -r result/
# mkdir result
# mobilenet_v3_large resnet_18 googlenet shufflenet_v2 inception_v3
# python validate.py --model mobilenet_v3_large --data /home/xiechenjia/imagenet
# python validate.py --model resnet_18 --data /home/xiechenjia/imagenet
# python validate.py --model resnet_50 --data /home/xiechenjia/imagenet
# python validate.py --model resnext_101 --data /home/xiechenjia/imagenet
# python validate.py --model googlenet --data /home/xiechenjia/imagenet
# python validate.py --model shufflenet_v2 --data /home/xiechenjia/imagenet
python validate.py --model inception_v3 --data /home/xiechenjia/imagenet
# python validate.py --model mobilenet_v2 --data /home/xiechenjia/imagenet
# python validate.py --model ghostnet --data /home/xiechenjia/imagenet --actbit 8
# python comp_rate.py 

