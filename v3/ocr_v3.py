import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import torchvision.datasets as datasets

from utils import *

characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

width, height, n_len, n_classes =160, 60, 4, len(characters)

num_workers=8

# 数据预处理和加载，也许可以增强一下
transform = transforms.Compose([
    transforms.Grayscale(),
    # transforms.RandomRotation(10),
    # transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Resize((60, 160)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])


dataset_path = 'dataset'
train_path = os.path.join(dataset_path, 'train')
valid_path = os.path.join(dataset_path, 'valid')
test_path = os.path.join(dataset_path, 'test')

#训练集
batch_size=64

class ImageFolderWithFoldername(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithFoldername, self).__getitem__(index)
        path, _ = self.samples[index]
        foldername = os.path.basename(os.path.dirname(path))
        return original_tuple[0], encode(foldername).view(1,-1)[0]


class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 20 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  
            nn.Linear(1024, 4 * n_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

if __name__=="__main__":
    
    
    train_dataset = ImageFolderWithFoldername(train_path, transform=transform)
    train_loader =DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    # #验证集
    valid_dataset = ImageFolderWithFoldername(valid_path, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    #测试集
    test_dataset=ImageFolderWithFoldername(test_path,transform=transform)
    test_loader =DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    #训练模型
    model = CNN(n_classes=n_classes)
    model=model.to(device)

    criterion = nn.MultiLabelSoftMarginLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        # 训练模型
        train_loss = train(model, optimizer, train_loader, device, criterion)
        # 在验证集上评估模型
        val_loss, accuracy = valid(model, valid_loader, criterion, device)
 
        # 打印当前训练和验证集的损失和准确率
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")

                
    torch.save(model, 'captcha_model.pth')

    model=torch.load('captcha_model.pth').to(device)

    test(model,test_loader,device)