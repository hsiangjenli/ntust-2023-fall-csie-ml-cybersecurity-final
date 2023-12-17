from model import CNNModel
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import shap
import os
import pickle
from utils import *
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--noise_type', type=str, default='normalize')
parser.add_argument('--image_dir', type=str, default='data/binary2image')
parser.add_argument('--output_dir', type=str, default='data/noise')

args = parser.parse_args()

# -- Setup paths ---------------------------------------------------------------
test_csv = "data/red/test.csv"
model_path = "bin/red_team.pkl"
# ------------------------------------------------------------------------------

transforms_step = [
    transforms.Resize((128, 128)),
    transforms.ToTensor()
]

if args.noise_type == 'normalize':
    transforms_step.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

test_transform = transforms.Compose(transforms_step)

model = CNNModel(num_classes=6)
model.load_state_dict(torch.load(model_path))

model.eval()

ImageDataset = ImageDataset(args.image_dir, pd.read_csv(test_csv), test_transform)
test_loader = DataLoader(ImageDataset, batch_size=100, num_workers=8)

batch = next(iter(test_loader))
images, labels = batch

background = images
test_images = images

e = shap.GradientExplainer(model, background)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

print(len(shap_numpy))
print(len(test_numpy[0]))

os.makedirs(f"{args.output_dir}/{args.noise_type}", exist_ok=True)

for c in range(len(shap_numpy)):
    for i in range(len(shap_numpy[c])):
        original_image = test_numpy[i]
        pos_contrib = shap_numpy[c][i] >  0
        
        mask = np.zeros_like(original_image)
        mask[pos_contrib] = original_image[pos_contrib]

        _class_name = list(lable_dic)[c]

        pickle.dump(mask, open(f"{args.output_dir}/{args.noise_type}/{_class_name}_{i}.pkl", 'wb'))

pickle.dump(shap_numpy, open(f"{args.output_dir}/{args.noise_type}_noise.pkl", "wb"))
shap.image_plot(shap_numpy, test_numpy)
plt.show()
plt.savefig('shap.png')