# 验证码识别

## 📖 介绍

这是一个基于机器学习的验证码识别python实现，分为2个版本，分别使用了sklearn库的SVM模型和pytorch的CNN,其中pytorch可以使用cuda进行加速。v1没啥用，v2和v3训练集为3w张图片时大概能有70-80%正确率。能不能跑，谁知道呢

## 💿 依赖

可能需要额外安装的依赖详情如下：

### V1

需要将OCR_Dataset.rar解压

```python
numpy
opencv#cv2
scikit-learn#sklearn
```

### V2

>设备可能需要安装有CUDA

```python
numpy
torch
captcha#验证码库
tqdm#进度条库
```

## 🎉 运行

V1:
直接运行`ocr_v1.py`即可，程序会自动训练并保存模型，并输出模型准确度的评估结果
V2:
先调整`captcha_generator.py`的参数，然后运行它生成数据集，最后运行`ocr_v2.py`
V3:
先调整`captcha_generator.py`的参数，然后运行它生成数据集，最后运行`ocr_v2.py`(和v2一样)
