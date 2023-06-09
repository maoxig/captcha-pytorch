import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle

TRAIN_SIZE = 0.2

# 需要将OCR_Dataset.rar解压到当前目录
labels = {}
with open("label_dict.txt", "r") as f:
    content = eval(f.read())
    labels = content


def extract_features(img_data):
    vertical_projection = np.sum(img_data, axis=0)
    horizontal_projection = np.sum(img_data, axis=1)
    return np.concatenate((vertical_projection, horizontal_projection))


def preprocess_img(img):
    """获取图片数据，返回四个字符图片矩阵的列表"""
    X = []
    # 去除边框
    img_data = img[2:-2, 3:-3]
    # 灰度处理
    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    # 高斯模糊
    img_data = cv2.GaussianBlur(img_data, (3, 3), 0)
    # 二值化
    _, img_data = cv2.threshold(
        img_data, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # 切割为四分字符
    character_images = np.array_split(img_data, 4, axis=1)
    for _, character_image in enumerate(character_images):
        feature_vector = extract_features(character_image)
        X.append(feature_vector)
    return X


X = []
y = []
data = []
for filename in os.listdir("OCR_Dataset"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread("OCR_Dataset/" + filename)
        img_data = np.array(img)
        X.extend(preprocess_img(img_data))
        y.extend([int(x) for x in labels[filename]])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)
model = SVC(kernel="linear", C=1, probability=True, verbose=1)
model.fit(X_train, y_train)

# 保存模型
with open("model.pickle", "wb") as f:
    pickle.dump(model, f)
print("保存模型")
# 预测准确度
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("准确度为:", accuracy)

# #以下为测试demo
# test_img = cv2.imread("demo.jpg")
# test_img = np.array(test_img)
# test_img = preprocess_img(test_img)
# test_result = model.predict(test_img)
# print("demo结果为:", test_result)
