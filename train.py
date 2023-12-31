import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

from utils import *
from model import CNNModel
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
set_seed(42)

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str)
argparser.add_argument('--train_csv', type=str)
argparser.add_argument('--test_csv', type=str)
argparser.add_argument('--model_name', type=str)

args = argparser.parse_args()


train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(args.data_dir, pd.read_csv(args.train_csv), train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True, drop_last=True, pin_memory=True)

df = pd.read_csv(args.train_csv)
df["l"] = df["label"].map(lable_dic)
num_classes = len(df["label"].unique())

model = CNNModel(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

weight = df["l"].value_counts().sort_index().to_list()
weight = torch.Tensor([26/x for x in weight]).to(device)

criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.15)
optimizer = torch.optim.RAdam(model.parameters(), lr=0.0003)

accuracy_fn = Accuracy(task="MULTICLASS",num_classes=num_classes).to(device)
F1Score_fn = F1Score(task="MULTICLASS",num_classes=num_classes).to(device)
best_F1 = 0
patience = 10  # 10個 epoch 為上限
best_loss = None # 紀錄最好的loss
patience_count = 0 # 計數
# Training loop
for epoch in range(100):
    train_losses, train_acc, train_F1 = 0, 0, 0
    model.train()

    for index, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)
        out = model(images)

        loss = criterion(out, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_acc += accuracy_fn(out.argmax(dim=1), labels)
        train_F1 += F1Score_fn(out.argmax(dim=1), labels)
        train_losses += loss

    train_losses /= len(train_loader)
    train_acc /= len(train_loader)
    train_F1 /= len(train_loader)

    print(f"Epoch: {epoch}, Loss: {train_losses}, Train Accuracy: {train_acc}, Train Macro-F1: {train_F1}")

    if best_loss is None:
        best_loss = train_losses
    elif train_losses < best_loss:
        best_loss = train_losses
        patience_count = 0
    else:
        patience_count += 1

    if patience_count >= patience:
        print(f"Early stopping on epoch {epoch}")
        break

    if train_F1 > best_F1:
        best_F1 = train_F1

        torch.save(model.state_dict(), f'/workspace/bin/{args.model_name}.pkl')