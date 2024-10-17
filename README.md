# MultiModal001_CLIP
MultiModal001: CLIP using pytorch

## 使用说明
### 要求
> Python >= 3.6 \
> PyTorch >= 1.7.0  
### 预训练模型
```shell script
mkdir weights
cd weights
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```
### 测试
```shell script
python predict.py  
```
## 参考
https://github.com/OpenAI/CLIP   
https://blog.csdn.net/samylee  
