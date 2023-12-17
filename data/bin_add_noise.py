import argparse
import subprocess

parser = argparse.ArgumentParser(description='Add noise to binary files')
parser.add_argument('--vbinary_input_folder', type=str, default="raw", help='Folder containing vbinary')
parser.add_argument('--vbinary_output_folder', type=str, default="vbinary", help='Folder containing vbinary')
parser.add_argument('--test_csv', type=str, default=True)
parser.add_argument('--noise_input_folder', type=str, default="noise/original", help='Folder containing noise')
parser.add_argument('--noise_class', type=str, default="icedid", help='Class of noise to add')
parser.add_argument('--noise_sample_num', type=int, default=10, help='Sample number of noise to add')
parser.add_argument('--noise_layer', type=int, default=1, help='Add how many layers of noise')

args = parser.parse_args()

def get_binary_last_line(file_path):
    line = subprocess.check_output(['tail', '-1', file_path])
    return line

def open_binary(file_path):
    with open(file_path, "rb") as f:
        binary = f.read()
    return binary

def load_noise(input_folder, noise_class, sample_num):
    noise = open_binary(f"{input_folder}/{noise_class}_{sample_num}.pkl")
    binary_string = ''.join(format(byte, '08b') for byte in noise)
    binary_string = binary_string.encode('utf-8')
    binary_groups = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]

    return binary_groups

if __name__ == "__main__":
    import os
    import pandas as pd

    class_noise = load_noise(args.noise_input_folder, args.noise_class, args.noise_sample_num)
    os.makedirs(f"{args.vbinary_output_folder}/{args.noise_class}_{args.noise_layer}", exist_ok=True)

    if args.test_csv:
        file_names = [file for file in os.listdir(args.vbinary_input_folder) if not file.endswith('.keep')]
    else:
        test_df = pd.read_csv(args.test_csv)
        file_names = test_df["Img"].tolist()

    for file_name in file_names:

        print(f"Processing {file_name}")

        file = open_binary(f"{args.vbinary_input_folder}/{file_name}")
        last_line = get_binary_last_line(f"{args.vbinary_input_folder}/{file_name}")[:8]

        with open(f"{args.vbinary_output_folder}/{args.noise_class}_{args.noise_layer}/{file_name}", "wb") as f:
            f.write(file)

            for _ in range(args.noise_layer):
                for i, group in enumerate(class_noise):
                    if i % 6 == 0:
                        
                        if i != 0:
                            f.write(b'  ......')
                            f.write(b"\n")
                        
                        decimal_value = int(last_line, 16)
                        
                        offset = decimal_value + i
                        f.write(b"00")
                        f.write(hex(offset)[2:].encode('utf-8'))
                        f.write(b":")

                    
                    f.write(b" ")
                    f.write(group)