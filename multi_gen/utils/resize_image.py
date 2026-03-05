from PIL import Image
import glob
import os


# if __name__ == "__main__":
#     input_dir = "data_bird/bird_12345"
#     output_dir = "data_bird/bird_12345_resize"

#     os.makedirs(output_dir, exist_ok=True)
#     image_list = glob.glob(os.path.join(input_dir, "*.png"))

#     for idx, image_path in enumerate(image_list):
#         image_name = os.path.basename(image_path)
#         img = Image.open(image_path)
#         w, h = img.size

#         img_out = img.resize((w * 2, h * 2), Image.BICUBIC)

#         output_path = os.path.join(output_dir, image_name)
#         img_out.save(output_path)


if __name__ == "__main__":
    input_dir = "data_bird/bird_12345_resize_sketch/image*.png/mask.png"

    image_list = glob.glob(input_dir)

    for idx, image_path in enumerate(image_list):
        image_name = os.path.basename(image_path)
        img = Image.open(image_path)
        w, h = img.size
        target_w = target_h = 512

        img_out = img.resize((target_w, target_h), Image.BICUBIC)

        output_path = image_path.replace("mask.png", "mask_resize.png")
        img_out.save(output_path)
