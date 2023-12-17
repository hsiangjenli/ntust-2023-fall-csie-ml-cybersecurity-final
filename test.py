import argparse
from model import CNNModel
from utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import ttach as tta
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str)
argparser.add_argument('--test_csv', type=str)
argparser.add_argument('--model_name', type=str)

args = argparser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv(args.test_csv)
num_classes = test_df["label"].nunique()

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(20),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tta_transforms = tta.Compose([
    tta.HorizontalFlip(),
])

test_dataset = ImageDataset(args.data_dir, test_df, test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, drop_last=True, pin_memory=True)

model = CNNModel(num_classes)

weight = test_df.label.value_counts().sort_index().tolist()
weight = torch.Tensor([26/x for x in weight]).to(device)

criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.15)
accuracy_fn = Accuracy(task="MULTICLASS",num_classes=num_classes).to(device)
F1Score_fn = F1Score(task="MULTICLASS",num_classes=num_classes).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(f"bin/{args.model_name}.pkl"))
tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
test_loss, test_acc, test_F1 = 0, 0, 0

model.eval()

if "_" in args.data_dir:
   noise_class = args.data_dir.split("/")[-1]
   noise_class, noise_layers = noise_class.split("_")[0], noise_class.split("_")[1]
else:
   noise_class = "without_noise"
   noise_layers = "without_noise"

output = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
      images, labels = images.to(device), labels.to(device)
      out = model(images)
      predictions = out.argmax(dim=1)

      for img, pred, label in zip(test_loader.dataset.df['Img'], predictions.cpu().numpy(), labels.cpu().numpy()):
          predicted_label = list(lable_dic.keys())[list(lable_dic.values()).index(pred)]
          actual_label = list(lable_dic.keys())[list(lable_dic.values()).index(label)]
          img = img.split("/")[-1]
          output.append({"noise_class": noise_class, "img": img, "predicted_label": predicted_label, "actual_label": actual_label})
          print(f"圖片: {img}, 模型預測類別: {predicted_label}, 真實類別: {actual_label}")
      
      t_loss = criterion(out, labels)
      test_acc += accuracy_fn(out.argmax(dim=1), labels)
      test_F1 += F1Score_fn(out.argmax(dim=1), labels)
      test_loss += t_loss

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    test_F1 /= len(test_loader)
    
    print(f"\nLoss: {test_loss}, Test Accuracy: {test_acc}, Test Macro-F1: {test_F1}")

output_df = pd.DataFrame(output)
output_df.to_csv(f"score/{args.data_dir.split('/')[-1]}.csv", index=False)