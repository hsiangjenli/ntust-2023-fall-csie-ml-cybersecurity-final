import pickle
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Add noise to binary files')
parser.add_argument('--noise_class', type=int, default=4, help='Class of noise to add')
parser.add_argument('--noise_sample_num', type=int, default=10, help='Sample number of noise to add')
parser.add_argument('--noise_file', type=str, default="/workspace/bin/shap_values.pkl", help='File containing noise')
parser.add_argument('--noise_layer', type=int, default=1, help='Add how many layers of noise')

args = parser.parse_args()

def get_binary_last_line(file_path):
    line = subprocess.check_output(['tail', '-1', file_path])
    return line

def open_binary(file_path):
    with open(file_path, "rb") as f:
        binary = f.read()
    return binary

def load_noise(file_path, num_class, num_sample=10):
    noise = pickle.load(open(file_path, "rb"))
    noise = noise[num_class][num_sample].tobytes()
    binary_string = ''.join(format(byte, '08b') for byte in noise)
    utf8_bytes = binary_string.encode('utf-8')
    return utf8_bytes

if __name__ == "__main__":
    import os
    from io import BytesIO
    import struct
    from bitstring import BitArray

    # Create the output folder --------------------------------------------------------------------------------
    output_folder = f"/workspace/data/noise_{args.noise_class}_layer_{args.noise_layer}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the noise -----------------------------------------------------------------------------------------
    noise = load_noise(args.noise_file, args.noise_class, args.noise_sample_num)
    noise_groups = [noise[i:i+8] for i in range(0, len(noise), 8)]

    # Get all files in the raw folder --------------------------------------------------------------------------
    file_paths = os.listdir("/workspace/data/raw")

    # Loop through all files in the raw folder -----------------------------------------------------------------
    for file_path in file_paths:
        file = open_binary(f"/workspace/data/raw/{file_path}")
        last_line = get_binary_last_line(f"/workspace/data/raw/{file_path}")[:8]

        print(f"Processing {file_path}")
        
        # Write the file to the noise folder -------------------------------------------------------------------
        with open(f"{output_folder}/{file_path}", "wb") as f:
            f.write(file)

            # Add noise to the file -----------------------------------------------------------------------------
            for _ in range(args.noise_layer):
                for i, group in enumerate(noise_groups):

                    if i % 6 == 0:
                        
                        if i != 0:
                            f.write(b'  ......')
                            f.write(b"\n")
                        
                        decimal_value = int(last_line, 16)
                        
                        offset = decimal_value + i
                        f.write(b"00")
                        f.write(hex(offset)[2:].encode('utf-8'))
                        f.write(b":")

                    # if (i+1) % 7 == 0:
                    #     f.write(" ......\n".encode("utf-8"))
                    
                    f.write(b" ")
                    f.write(group)