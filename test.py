
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from torchvision import transforms
import ttach as tta
from tqdm import tqdm
from utils import *
from model import CNNModel
import pickle
import sys
import random

# -- Setup training and test directories and csv files -------------------------
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test.csv'
# ------------------------------------------------------------------------------
def cal_img_width(file_size_kb):
    if file_size_kb < 10:
        img_width = 32
    elif 10 <= file_size_kb < 30:
        img_width = 64
    elif 30 <= file_size_kb < 60:
        img_width = 128
    elif 60 <= file_size_kb < 100:
        img_width = 256
    elif 100 <= file_size_kb < 200:
        img_width = 384
    elif 200 <= file_size_kb < 500:
        img_width = 512
    elif 500 <= file_size_kb < 1000:
        img_width = 768
    else:
        img_width = 1024
    return img_width

def img_2_binary(img):
    img = np.array(img)
    shape = img.shape
    img = img.tobytes()
    return img, shape

def load_noise(file_path, num_class):
    noise = pickle.load(open(file_path, "rb"))
    return noise[num_class][10]

def append_noise(img):

    noise = load_noise("/workspace/bin/shap_values.pkl", 4)

    img, shape = img_2_binary(img)
    
    noise, _ = img_2_binary(noise)
    img_noise = img# + noise

    img_width = shape[1]
    img_height = -(-len(img_noise) // img_width)

    # 使用 pad 補滿
    padded_data = np.frombuffer(img_noise, dtype=np.uint8)
    padded_data = np.pad(padded_data, (0, img_width * img_height - len(img_noise)), 'constant', constant_values=0)

    reshaped_data = padded_data.reshape((img_height, img_width)) 
    image = Image.fromarray(reshaped_data, 'L')
    image = image.convert('RGB')

    image.save(f"data/test_noise_4/{random.random()}.png")

    return image

# ------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Lambda(append_noise),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tta_transforms = tta.Compose([
    tta.HorizontalFlip(),
])
test_dataset = ImageDataset(TEST_DIR, pd.read_csv(TEST_CSV), test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, drop_last=True, pin_memory=True)

model = CNNModel()

weight = [7,62,23,16,24]
weight = torch.Tensor([26/x for x in weight]).to(device)

criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.15)
accuracy_fn = Accuracy(task="MULTICLASS",num_classes=5).to(device)
F1Score_fn = F1Score(task="MULTICLASS",num_classes=5).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("/workspace/bin/MalewareClassifier_bestF1.pkl"))
tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
test_loss, test_acc, test_F1 = 0, 0, 0

print(model)

model.eval()

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.cuda(), labels.cuda()
        out = model(images)
        t_loss = criterion(out, labels)
        test_acc += accuracy_fn(out.argmax(dim=1), labels)
        test_F1 += F1Score_fn(out.argmax(dim=1), labels)
        test_loss += t_loss

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    test_F1 /= len(test_loader)
  
    print(f"\nLoss: {test_loss}, Test Accuracy: {test_acc}, Test Macro-F1: {test_F1}")