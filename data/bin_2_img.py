import re
import numpy as np
from PIL import Image
import os

def best_img_width(file_size_kb):
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

def binary_file_to_image(input_file_path, output_file_path):
    
    with open(input_file_path, 'r') as file:
        contents = file.read()

    binary_data = re.findall(r'[01]{8}', contents)
    grayscale_values = [int(b, 2) for b in binary_data]


    file_size_kb = os.path.getsize(input_file_path) / 1024  # KB
    img_width = best_img_width(file_size_kb)
    img_height = (len(grayscale_values) + img_width - 1) // img_width
    
    image_data = grayscale_values + [0] * (img_width * img_height - len(grayscale_values))
    np_image = np.array(image_data, dtype=np.uint8).reshape((img_height, img_width))
    image = Image.fromarray(np_image, 'L')

    try:

        image.save(output_file_path)
        print(f"Image saved to {output_file_path}")
    
    except Exception as e:
        print(e)
        print(f"Failed to save image to {output_file_path}")

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_folder')
    parser.add_argument('--output_file_folder')
    args = parser.parse_args()

    os.makedirs(args.output_file_folder, exist_ok=True)

    for file in os.listdir(args.input_file_folder):
        input_file_path = os.path.join(args.input_file_folder, file)
        output_file_path = os.path.join(args.output_file_folder, file.split('.')[0] + '.png')
        binary_file_to_image(input_file_path, output_file_path)