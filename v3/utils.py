import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def train(model, optimizer, dataloader, device, criterion):
    model.train()  # 设置模型为训练模式
    train_loss = 0.0  # 初始化训练损失为0
    total_samples = 0  # 总样本数

    with tqdm(dataloader, desc="Training", leave=False) as pbar:  # 使用tqdm创建进度条
        for batch_index, (inputs, targets) in enumerate(pbar):  # 遍历数据加载器中的每个批次
            inputs = inputs.to(device)  # 将输入数据移动到指定的设备（如GPU）
            targets = targets.to(device)  # 将目标数据移动到指定的设备（如GPU）
            optimizer.zero_grad()  # 梯度清零，防止梯度累积
            outputs = model(inputs)  # 前向传播，获取模型的输出
            loss = criterion(outputs, targets)  # 计算损失值
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数，执行优化步骤
            train_loss += loss.item() * inputs.size(0)  # 累加训练损失值，乘以批次大小得到批次损失总和
            total_samples += inputs.size(0)  # 更新总样本数
            pbar.set_postfix({"Loss": loss.item()})  # 在进度条上显示当前批次的损失值
    train_loss /= total_samples  # 计算平均训练损失值
    return train_loss  # 返回平均训练损失值

def valid(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        model.to(device)
        total_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            targets = targets.view(-1, 4, 36)
            outputs = outputs.view(-1, 4, 36)
            targets = decode(targets)
            outputs = decode(outputs)

            batch_correct = count_matching_strings(targets, outputs)
            total_correct += batch_correct
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            progress_bar.set_postfix(
                {
                    "loss": total_loss / total_samples,
                    "accuracy": total_correct / total_samples,
                }
            )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


def count_matching_strings(list1, list2):
    count = 0
    for str1, str2 in zip(list1, list2):
        if str1 == str2:
            count += 1
    return count



def test(model, dataloader, device):
    model.eval()
    total_correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.view(-1, 4, 36)
            targets_string = decode(targets)
            outputs = model(inputs)
            outputs = outputs.view(-1, 4, 36)  # 调整output形状
            outputs_string = decode(outputs)
            if outputs_string == targets_string:
                total_correct += 1
                print(f"预测正确：正确值:{targets_string},预测值:{outputs_string}")
            else:
                print(f"预测失败:正确值:{targets_string},预测值:{outputs_string}")

        print(f"正确率{total_correct/len(dataloader)}")


def encode(text):
    """将字符编成onehot码"""
    vectors = torch.zeros((4, 36))
    for i in range(len(text)):
        vectors[i, characters.index(text[i])] = 1
    return vectors


def decode(prediction):
    batch_size = prediction.shape[0]
    decoded_labels = ["".join([characters[torch.argmax(prediction[i, j])] for j in range(4)]) for i in range(batch_size)]
    return decoded_labels

if __name__=="__main__":
    inputs=encode("0ABC")
    a="AABBCCDD"
    b="DDBBCCAA"

    print(inputs)
    ouputs=decode(inputs)
    print(ouputs)