import os

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights


def main():
    # ========================
    # 1. 參數設定
    # ========================
    data_dir = "data/plant-seedlings-classification"
    batch_size = 64
    num_classes = 12
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================
    # 2. 資料轉換 (Transforms)
    # ========================
    data_transforms = {
        "train": transforms.Compose([

            transforms.RandomResizedCrop(  # 隨機裁一塊原圖區域並縮放到指定大小(224*224)
                224,
                scale=(0.9, 1.0),  # 控制裁切區域最小佔原圖 90%
                ratio=(0.9, 1.1),  # 控制裁切區域長寬比落在 0.9~1.1
                antialias=True  # 建議 torchvision>=0.13 加這個
            ),
            transforms.RandomHorizontalFlip(p=0.5),  # 50% 機率水平翻轉
            transforms.ColorJitter(
                brightness=0.2,  # 亮度 ±20%
                contrast=0.2,  # 對比度 ±20%
                saturation=0.2,  # 飽和度 ±20%
                hue=0.05  # 色相偏移 ±5%
            ),
            transforms.RandomRotation(15),  # 隨機旋轉 ±15 度
            transforms.RandomVerticalFlip(p=0.5),  # 50% 機率垂直翻轉
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            ], p=0.5),  # 50% 機率啟用
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.5,  # 50% 機率執行
                scale=(0.02, 0.1),  # 擦除區域佔原圖比例範圍 2%~10%
                ratio=(0.3, 3.3),  # 擦除區域長寬比範圍
                value='random'  # 用隨機值填充擦除區域
            )
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 從中心裁切成 224*224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # ========================
    # 3. 建立 Dataset & DataLoader
    # ========================
    train_dir = os.path.join(data_dir, "train")

    # 把 train 分成 train/val (8:2)
    dataset = datasets.ImageFolder(train_dir)  # 會自動根據子資料夾分不同類設定到 img、label，先不指定 transform，分割後再分別指定
    print(dataset.class_to_idx)  # 印出類別對應的 index
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 指定各自的 transform
    train_dataset.dataset.transform = data_transforms["train"]
    val_dataset.dataset.transform = data_transforms["val"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ========================
    # 4. 載入 ResNet50
    # ========================
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # 凍結所有層參數更新
    for param in model.parameters():
        param.requires_grad = False
    # 解凍 layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # ========================
    # 5. Loss & Optimizer
    # ========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每 5 個 epoch lr * 0.5

    # ========================
    # 6. 訓練
    # ========================
    # 記錄每個 epoch 的 loss 和 accuracy，用於畫圖
    train_losses, val_losses = [], []  # 畫圖用

    for epoch in range(num_epochs):
        # --------- 訓練 ---------
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()  # 更新 scheduler lr

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)

        # --------- 驗證 ---------
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        val_loss /= len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)
        val_losses.append(val_loss)

        # --------- 每 5 個 epoch 儲存一次模型 ---------
        if (epoch + 1) % 5 == 0:
            model_filename = f"epoch{epoch + 1}_valLoss_{val_loss:.4f}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"模型已保存：{model_filename}")

        print(f"Epoch {epoch + 1}/{num_epochs} "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # # ========================
    # # 7. 畫出 Loss 曲線
    # # ========================
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training & Validation Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # ========================
    # 8. 儲存模型
    # ========================
    torch.save(model.state_dict(), "resnet50_plant.pth")

    # =======================
    # 9. 推理 validation set，產生 confusion matrix
    # ========================
    # 建立相同的模型結構
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # 載入訓練好的權重
    model.load_state_dict(
        torch.load("resnet50_plant.pth", map_location=device,
                   weights_only=True))  # weights_only 是安全性設定
    model.to(device)
    model.eval()  # 設為推論模式

    all_preds = []
    all_labels = []
    with torch.no_grad():  # 推論不需要計算梯度
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)  # 模型輸出
            _, preds = torch.max(outputs, 1)  # 取最大值的 index

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 轉成一維 array
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 計算 confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    plt.figure(figsize=(15, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set)')
    plt.show()


if __name__ == "__main__":
    main()
