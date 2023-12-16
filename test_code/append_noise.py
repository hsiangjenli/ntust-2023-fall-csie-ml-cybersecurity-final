# Read Test Images and append noise to them

import numpy as np
import pickle
from PIL import Image
from torchvision import transforms

shap_value = pickle.load(open("/workspace/bin/shap_values.pkl", "rb"))

# -- Setup training and test directories and csv files -------------------------
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test.csv'
# ------------------------------------------------------------------------------

def img_2_binary(img):
    img = np.array(img)
    img = img.astype(np.uint8)
    return img

def append_noise(img, noise):
    img = img_2_binary(img)
    noise = img_2_binary(noise)
    img = img + noise
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

test_noise_transform = transforms.Compose([
    append_noise,
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

