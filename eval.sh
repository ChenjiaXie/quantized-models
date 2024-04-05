log_path=./LOGS
wt_path=./qmodels
### models: vgg_* resnet_* convnext_* mobilenet_* efficientnet_* inception_v3 shufflenet_* googlenet
model=convnext_tiny
data_path=/data-hdd/xiechenjia/DATASETS/CNN_DATASETS/imagenet-1k

mkdir -p $log_path
mkdir -p $wt_path
export TORCH_HOME=$wt_path


python validate.py --model $model --data $data_path > $log_path/$model.log