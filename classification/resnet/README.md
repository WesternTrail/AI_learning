## 用简单的二分类实现人脸识别
数据集采用的是imagenet的格式组织，分别为两类：马云和彭于晏。采用resnet34进行图片分类即可

## 文件结构：
```
  ├── model.py: ResNet模型搭建
  ├── train.py: 训练脚本
  ├── predict.py: 单张图像预测脚本
  └── batch_predict.py: 批量图像预测脚本
```