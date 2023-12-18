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
parser.add_argument('--test_csv', type=str, default='data/red/test.csv')
parser.add_argument('--output_dir', type=str, default='data/noise')
parser.add_argument('--model_name', type=str, default='red_team')
parser.add_argument('--save_noise', type=bool, default=True)

args = parser.parse_args()

model_path = f"bin/{args.model_name}.pkl"

num_classes = 6 if args.model_name == 'red_team' else 5

transforms_step = [
    transforms.Resize((128, 128)),
    transforms.ToTensor()
]

if args.noise_type == 'normalize':
    transforms_step.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

test_transform = transforms.Compose(transforms_step)

model = CNNModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))

model.eval()

ImageDataset = ImageDataset(args.image_dir, pd.read_csv(args.test_csv), test_transform)
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

        print(f"{_class_name}_{i}: total contrib: {np.sum(pos_contrib)}")

        if args.save_noise:
            pickle.dump(mask, open(f"{args.output_dir}/{args.noise_type}/{_class_name}_{i}.pkl", 'wb'))

shap.image_plot(shap_numpy, test_numpy)
plt.show()

prefix = args.image_dir.split('/')[-1]

plt.savefig(f'{args.model_name}_{prefix}_shap.png')