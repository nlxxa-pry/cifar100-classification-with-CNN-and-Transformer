# cifar100-classification-with-CNN-and-Transformer

本次实验使用CeiT作为 CNN + Transformer 的网络模型  
使用 ResNet50 作为相同参数量的CNN对比模型1  
使用 DenseNet201 作为相同浮点数计算量的CNN对比模型2 

## 文件下载
在这里下载checkpoint和log文件  
  
CeiT：  
百度网盘链接：https://pan.baidu.com/s/1zvmNg50kYrgA9nL-8d2myg  
提取码：4aej  
  
CNN：  
百度网盘链接：https://pan.baidu.com/s/1nIJchHSRkY14v3aSb7xG3w  
提取码：bt6x  


## CNN
CNN 文件夹中包含两个模型，ResNet50 和 DenseNet201 
 
### train:
```python
python train.py -net resnet50/densenet201 -gpu
```

### test：
test.py 文件位于CNN/utils/中
```python
python test.py -net resnet50/densenet201  -weights checkpoint_path -gpu
```
checkpoint_path 是已训练好的模型的路径，如checkpoint/densenet201.pth
 
 
## CeiT
CNN_Transformer 文件夹中包含CeiT的相关文件 


### Train :
```
python train.py -c configs/default.yaml --name "name_of_exp"
```

### Usage :
```python
import torch
from ceit import CeiT

img = torch.ones([1, 3, 224, 224])
    
model = CeiT(image_size = 224, patch_size = 4, num_classes = 100)
out = model(img)

print("Shape of out :", out.shape)      # [B, num_classes]

model = CeiT(image_size = 224, patch_size = 4, num_classes = 100, with_lca = True)
out = model(img)

print("Shape of out :", out.shape)      # [B, num_classes]

```




## Reference:
* [Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/pdf/2103.11816v2.pdf)
* https://github.com/rishikksh20/CeiT-pytorch.git
* https://github.com/weiaicunzai/pytorch-cifar100.git
