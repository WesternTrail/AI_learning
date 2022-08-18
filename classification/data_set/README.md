# 该文件夹用于存放训练数据
## 使用步骤如下：
- （1）在data_set文件夹下创建新文件夹"face_data"
- （2）点击链接下载人脸数据集 链接：https://pan.baidu.com/s/1fYp2y7fNnIrZn4Ip1HDYMg 
提取码：3uts
- （3）解压数据集到face_data文件夹下
- （4）将split_data.py放在data_set根目录下，执行"split_data.py"脚本自动将数据集划分成训练集train和验证集val

## data_set组织如下如下所示：

```
├─face_data
│  ├─face_photos
│  │  ├─mayun
│  │  └─pengyuyan
│  ├─train
│  │  ├─mayun
│  │  └─pengyuyan
│  └─val
│      ├─mayun
│      └─pengyuyan
└──split_data.py
```
