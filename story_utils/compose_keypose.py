import os
import numpy as np
from PIL import Image
import cv2
import math
import argparse
from omegaconf import OmegaConf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to compose config file", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    config = OmegaConf.load(args.config)
    # height, width = 512, config.image_width

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    input_subject_dict = [v for k, v in config.items() if k.startswith("layout")]

    # draw the subject
    for ind, sub in enumerate(input_subject_dict):
        # create a black image
        width = sub.get("width",1024)
        height = sub.get("height",512)
        img = np.zeros((height, width, 3), np.uint8)
        subject_path_list = sub["keypose_path"]
        box_layout_list = sub["box_layout"]
        for subject_path, box_layout in zip(subject_path_list, box_layout_list):
            # get the subject size
            x0, y0, w, h = box_layout
            x1 = x0 + w
            y1 = y0 + h
            # use ceil for x0 and y0, following mixofshow
            # x0, y0, x1, y1 = math.ceil(x0 * width), math.ceil(y0 * height), int(x1 * width), int(y1 * height)
            subject_width = x1 - x0
            subject_height = y1 - y0
            # load image with PIL
            print(subject_width, subject_height)
            subject_img = Image.open(subject_path)
            subject_img = subject_img.resize((subject_width, subject_height))  # TODO we should keep aspect ratio
            subject_img = np.array(subject_img)

            # draw the subject
            img[y0:y1, x0:x1] = subject_img

        img = Image.fromarray(img)
        output_path = os.path.join(output_dir, sub["output_image"])
        img.save(output_path)
