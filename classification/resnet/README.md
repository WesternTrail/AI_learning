## 用简单的二分类实现人脸识别
数据集采用的是imagenet的格式组织，分别为两类：马云和彭于晏。采用resnet34进行图片分类即可

## 文件结构：
```
  ├── model.py: ResNet模型搭建
  ├── train.py: 训练脚本
  ├── predict.py: 单张图像预测脚本
  └── batch_predict.py: 批量图像预测脚本
```

## 如何修改自己的分类类别
"train.py中"第71行的```net.fc = nn.Linear(in_channel, 2)```2改成你自己的类别，"predict.py"中的第39行
```model = resnet34(num_classes=2).to(device)```num_classes改成自己的类别