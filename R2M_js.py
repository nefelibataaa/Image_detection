import os
from PIL import Image
import numpy as np
import time

start_time = time.time()
def convert_to_binary(file_path):
    with open(file_path, 'rb') as f:
        return f.read()

def binary_to_grayscale_image(binary_data, image_size):
    data_len = len(binary_data)
    side_length = int(np.ceil(np.sqrt(data_len)))
    padded_data = binary_data.ljust(side_length * side_length, b'\0')

    img_data = np.frombuffer(padded_data, dtype=np.uint8)
    img_data = img_data.reshape((side_length, side_length))

    img = Image.fromarray(img_data, 'L')
    img = img.resize((image_size, image_size), Image.LANCZOS)
    return img

def process_directory(input_dir, output_dir, image_size=64):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.js'):
            file_path = os.path.join(input_dir, file_name)
            binary_data = convert_to_binary(file_path)
            img = binary_to_grayscale_image(binary_data, image_size)
            img.save(os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}.png'))

def main():
    benign_dir = "E:/python_project/Dataset/JavaScript源代码/js_benign"
    malicious_dir = "E:/python_project/Dataset/JavaScript源代码/js_malicious"
    output_benign_dir = "E:/python_project/Image_detection/image_benign_js"
    output_malicious_dir = "E:/python_project/Image_detection/image_malicious_js"

    process_directory(benign_dir, output_benign_dir)
    process_directory(malicious_dir, output_malicious_dir)

if __name__ == '__main__':
    main()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total Execution Time: {elapsed_time} seconds")