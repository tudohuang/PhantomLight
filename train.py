import glob
import imageio.v2 as imageio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import albumentations as A
import csv
import os
import random
import argparse

# 定義CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = self.fc1(x.view(x.size(0), -1))
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 獲取標籤的函數
def get_label(f):
    # 根據檔案名稱獲取標籤
    if 'incendio' in f.lower():
        label = 1
    elif 'aqua' in f.lower():
        label = 2
    elif 'arresto' in f.lower():
        label = 3
    elif 'alohomora' in f.lower():
        label = 4
    elif 'lumos' in f.lower():
        label = 5
    elif 'null' in f.lower():
        label = 0
    return label


# 訓練或驗證一個epoch的函數
def run_epoch(data, model, criterion, optimizer, device, is_train=True):
    total_loss = 0
    count = 0

    for dd in tqdm.tqdm(data):
        im, label = dd

        if is_train:
            transform = A.Compose([
              # A.Resize(28, 28),
              A.ShiftScaleRotate(p=0.5),
              A.OpticalDistortion(p=0.5),
              A.GridDistortion(p=0.5),
          ])
            im = transform(image=im)['image']

        im_d = torch.from_numpy(im[None, ...][None, ...]).to(device).float() / 255
        label_d = torch.from_numpy(np.array([label])).to(device)

        output_d = model(im_d)
        loss = criterion(output_d, label_d.long())

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count if count > 0 else 0

# 解析命令行參數
parser = argparse.ArgumentParser(description='Train a CNN for spell recognition')
parser.add_argument('--train_path', type=str, required=True, help='Path to the training files')
parser.add_argument('--model_path', type=str, default='.', help='Path to save the trained models')
args = parser.parse_args()

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取訓練數據
ffs = glob.glob(f'{args.train_path}/*/*.png')
train_list = []
for f in tqdm.tqdm(ffs):
    im = imageio.imread(f)
    label = get_label(f)
    train_list.append([im, label])

# 初始化模型、損失函數和優化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 設定訓練參數
epochs = 1000
csv_filename = 'training_results.csv'
header = ['Epoch', 'Training Loss', 'Validation Loss']

# 拆分訓練和驗證集
validation_split = 0.05
split_idx = int(len(train_list) * (1 - validation_split))
random.shuffle(train_list)
train_data, validation_data = train_list[:split_idx], train_list[split_idx:]

# 寫入CSV標頭
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

# 訓練循環
for epoch in range(epochs):
    model.train()
    random.shuffle(train_data)
    train_loss = run_epoch(train_data, model, criterion, optimizer, device, is_train=True)

    model.eval()
    with torch.no_grad():
        validation_loss = run_epoch(validation_data, model, criterion, optimizer, device, is_train=False)

    print(f"Epoch {epoch+1}, Training Loss: {train_loss:.5f}, Validation Loss: {validation_loss:.5f}")

    # 定期保存模型
    if epoch % 100 == 0:
        
        torch.save(model, f'{args.model_path}/model_epoch_{epoch}.pt')
        print(f"Model saved at epoch {epoch}")

# 保存最終模型
torch.save(model, f'{args.model_path}/final_model.pt')
print("Final model saved.")
