
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

# -- Setup training and test directories and csv files -------------------------
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test.csv'
# ------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
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