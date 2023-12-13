from model import CNNModel
from utils import *
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import shap
import matplotlib.pyplot as plt
import numpy as np

# -- Setup training and test directories and csv files -------------------------
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test.csv'
# ------------------------------------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = CNNModel()
model.load_state_dict(torch.load("/workspace/bin/MalewareClassifier_bestF1.pkl"))
model.eval()

# print(model.conv1_1)
test_dataset = ImageDataset(TEST_DIR, pd.read_csv(TEST_CSV), test_transform)
test_loader = DataLoader(test_dataset, batch_size=100, num_workers=8)

batch = next(iter(test_loader))
images, labels = batch

background = images
test_images = images

print(labels[:])

e = shap.GradientExplainer(model, background)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, test_numpy)
plt.savefig('shap.png')