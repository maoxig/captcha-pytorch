import os
import random
import shutil
from captcha.image import ImageCaptcha

# 数据集保存路径
dataset_path = 'dataset'
train_path = os.path.join(dataset_path, 'train')
valid_path = os.path.join(dataset_path, 'valid')
test_path = os.path.join(dataset_path, 'test')
# 生成训练集、验证集和测试集
num_train_samples = 12000   # 训练集样本数量
num_valid_samples = 2000   # 验证集样本数量
num_test_samples = 2000    # 测试集样本数量

# 创建保存数据集的文件夹
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 验证码字符集
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# 验证码生成器
captcha_generator = ImageCaptcha()

# 生成指定数量的验证码图像并保存到训练集、验证集和测试集文件夹
def generate_captcha_dataset(num_samples, save_path):
    for i in range(num_samples):
      # 随机生成验证码
        captcha_text = ''.join(random.choices(characters, k=4))
        captcha_image = captcha_generator.generate(captcha_text)

        # 生成图像文件名
        image_file = os.path.join(save_path, f'{captcha_text}', f'{captcha_text}.png')

        # 确保类别文件夹存在
        os.makedirs(os.path.join(save_path, f'{captcha_text}'), exist_ok=True)

        # 保存验证码图像
        with open(image_file, 'wb') as f:
            f.write(captcha_image.read())

        print(f'Saved image: {image_file}')


# 清空已存在的数据集文件夹
shutil.rmtree(train_path)
shutil.rmtree(valid_path)
shutil.rmtree(test_path)

# 创建新的数据集文件夹
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

generate_captcha_dataset(num_train_samples, train_path)
generate_captcha_dataset(num_valid_samples, valid_path)
generate_captcha_dataset(num_test_samples, test_path)
