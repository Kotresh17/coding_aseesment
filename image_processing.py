import cv2
import numpy as np
import os
from multiprocessing import Pool
import argparse


def get_mask(file_path, output_dir):
    """
    Function for counting white pixels
    file_path : input image path
    output_dir: out directory for saing the images
    """
    img = cv2.imread(file_path)
    if img is None:
        print(f"unable to read image: {file_path}")
        return 0
    mask = cv2.inRange(img, (200, 200, 200), (255, 255, 255))
    pixel_count = np.count_nonzero(mask)
    filename = os.path.basename(file_path)
    print(f" file_name : {filename}, white_pixel_count : {pixel_count} ")
    mask_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.png")
    cv2.imwrite(mask_filename, mask)
    
    return pixel_count

def main():
    parser = argparse.ArgumentParser(description="Provide images to create binary masks and count white pixels.")
    parser.add_argument("--input_dir", type=str, help="Path to input images")
    parser.add_argument("--output_dir", type=str, help="Path to output mask images")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.path.abspath(os.path.join(os.path.dirname(__file__), args.input_dir))
    image_files = [os.path.join(args.input_dir, image_file) for image_file in os.listdir(os.path.join(args.input_dir))]

    with Pool() as pool:
        white_pixel_counts = pool.starmap(get_mask, [(file, args.output_dir) for file in image_files])
    
    all_white_pixels = sum(white_pixel_counts)
    
    print(f"Total white pixels: {all_white_pixels}")


if __name__ == "__main__":
    main()
