import os
import ast
import math
import numpy as np
import cv2
import argparse
from omegaconf import OmegaConf
import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, help="path to the layout text",
                        default="output_stories/gpt-4_1024x512/Write_a_short_story_between_Tezuka_Kunimitsu_and_Hina_Amano./replicate_0")
    parser.add_argument("--negative_prompt", type=str, help="shared negative prompts",
                        default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
    parser.add_argument("--image_width", type=int, default=1024, help="image width")
    parser.add_argument("--adaptor_weight", type=float, default=1., help="strength of the adaptor")
    parser.add_argument("--update_keypose_compose_config", type=int, default=1, choices=[0, 1],
                        help="whether to update the keypose compose config")
    parser.add_argument("--update_final_gen_config", type=int, default=1, choices=[0, 1],
                        help="whether to update the final gen config")
    parser.add_argument("--cond_type", type=str, required=True, help="keypose condition type",
                        choices=["keypose", "sketch"], )
    args = parser.parse_args()
    return args


def parse_chatgpt(text=None, bg_prompt_text="Background prompt: "):
    caption = None
    if "Caption:" in text:
        caption = text.split("Objects: ")[0].split("Caption: ")[1].strip()
    if "Objects: " in text:
        text = text.split("Objects: ")[1]

    text_split = text.split(bg_prompt_text)

    if len(text_split) == 2:
        gen_boxes, bg_prompt = text_split
        if len(gen_boxes.split("\n")) != 1:
            gen_boxes = gen_boxes.split("\n")[0]
        gen_boxes = ast.literal_eval(gen_boxes) 
            
        bg_prompt = bg_prompt.strip()

    return gen_boxes, bg_prompt, caption


def box_yxyx_to_xywh(gen_boxes):
    phrases = []
    locations = []
    for box in gen_boxes:
        y0, x0, y1, x1 = box[1]
        w = x1 - x0
        h = y1 - y0
        box_out = [x0, y0, w, h]
        print(box_out)  # xyxy
        phrases.append(box[0])
        locations.append(box_out)

    return phrases, locations


def box_xywh_norm_to_xyxy_unorm_format(gen_boxes, image_width=1024, image_height=512):
    phrases = []
    locations = []
    for box in gen_boxes:
        x0, y0, w, h = box[1]
        box_gligen = [x0 / image_width, y0 / image_height, (x0 + w) / image_width, (y0 + h) / image_height]
        print(box_gligen)  # xyxy
        phrases.append(box[0])
        locations.append(box_gligen)

    return phrases, locations


def box_xywh_to_yxyx_format(gen_boxes):
    phrases = []
    locations = []
    for box in gen_boxes:
        x0, y0, w, h = box[1]
        box_mixofshow = [y0, x0, y0 + h, x0 + w]  # yxyx

        phrases.append(box[0])
        locations.append(box_mixofshow)

    return phrases, locations


def box_yxyx_unnorm_to_xyxy_norm_format(boxes, image_width=1024, image_height=512):
    locations = []
    for box in boxes:
        y0, x0, y1, x1 = box
        box_xyxy = [x0 / image_width, y0 / image_height, x1 / image_width, y1 / image_height]
        locations.append(box_xyxy)

    return locations


def read_layout_text_file(text_file_path):
    output_text_list = []
    output_one_layout = None
    # read the text file
    with open(text_file_path, 'r') as f:
        texts = f.readlines()
        idx = 0
        for text in texts:
            if text == '\n':
                continue
            idx += 1
            text = text.strip()

            if idx % 3 == 1:
                # print(text)
                assert text.startswith("Caption: ")
                if output_one_layout is not None:
                    output_text_list.append("\n".join(output_one_layout))
                output_one_layout = []
            elif idx % 3 == 2:
                assert text.startswith("Objects: ")
            elif idx % 3 == 0:
                assert text.startswith("Background prompt: ")
            
            output_one_layout.append(text)

        output_text_list.append("\n".join(output_one_layout))

    return output_text_list


if __name__ == "__main__":
    args = get_args()

    context_neg_prompt = args.negative_prompt
    image_width, image_height = args.image_width, 512

    # create output dir
    layout_dir = os.path.join(args.work_dir, "layouts")
    os.makedirs(layout_dir, exist_ok=True)
    compose_cond_dir = os.path.join(args.work_dir, f"composed_{args.cond_type}")
    os.makedirs(compose_cond_dir, exist_ok=True)

    # read the layout
    text_file_path = os.path.join(args.work_dir, "layout.txt")
    output_text_list = read_layout_text_file(text_file_path)

    global_prompt_list = []
    local_prompt_list = []
    bg_prompt_list = []
    neg_prompt_list = []
    panel_texts_list = []
    adaptor_weight_list = []
    box_xyxy_norm_list = []
    condtional_file_path_list = []
    for panel_idx, output_text in enumerate(output_text_list):
        gen_boxes, bg_prompt, caption = parse_chatgpt(output_text)  # gen_boxes generated by chatgpt are in xywh format, range 1024
        phrases, locations = box_xywh_to_yxyx_format(gen_boxes)  # mixofshow format is yxyx, range 1024
        locations_xyxy = box_yxyx_unnorm_to_xyxy_norm_format(locations, image_width=image_width, image_height=image_height)  # xyxy, range 0-1

        prompt_rewrite = ""
        for obj_idx, (phrase_, locat_) in enumerate(zip(phrases, locations)):
            if prompt_rewrite == "":
                prompt_rewrite = f"{phrase_}-*-{context_neg_prompt}-*-{locat_}"
            else:
                prompt_rewrite = prompt_rewrite + f"|{phrase_}-*-{context_neg_prompt}-*-{locat_}"

        local_prompt_list.append(phrases)
        box_xyxy_norm_list.append(locations_xyxy)
        global_prompt_list.append(caption)
        bg_prompt_list.append(bg_prompt)
        neg_prompt_list.append(context_neg_prompt)
        condtional_file_path_list.append(os.path.join(compose_cond_dir, f"layout_{args.cond_type}_{panel_idx}.png"))  # note these condition files are not created yet
        panel_texts_list.append(prompt_rewrite)
        adaptor_weight_list.append(args.adaptor_weight)

        # draw the layout
        image = np.zeros((image_height, image_width, 3), np.uint8)
        for (obj_name, box) in gen_boxes:
            x0, y0, w, h = box  # xywh
            box = [math.ceil(x0), math.ceil(y0), int(x0 + w), int(y0 + h)]  # xyxy
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            cv2.putText(image, obj_name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # save the image using cv2
        cv2.imwrite(os.path.join(layout_dir, f"layout_box_{panel_idx}.png"), image)

    assert len(global_prompt_list) == len(bg_prompt_list) == len(neg_prompt_list) == len(panel_texts_list) == len(adaptor_weight_list) == len(condtional_file_path_list) == len(box_xyxy_norm_list)


    if args.update_keypose_compose_config:
        # # save the config for compose condition
        # output_dir: output_stories/gpt-4_1024x512/Write_a_short_story_between_Tezuka_Kunimitsu_and_Hina_Amano./replicate_0/composed_keypose
        # image_width: 1024
        # layout1:
        #   output_image: layout_keypose_1.png
        #   box_layout:
        #   - box1
        #   - box2
        #   keypose_path:
        #   - keypose1_path
        #   - keypose2_path
        compose_condition_config = {
            "output_dir": compose_cond_dir,
            "image_width": image_width,
        }
        for obj_idx, (layout_cond_path, box_layout, local_prompt) in enumerate(zip(condtional_file_path_list, box_xyxy_norm_list, local_prompt_list)):
            this_layout = {
                f"layout{obj_idx}": {
                    "output_image": os.path.basename(layout_cond_path),
                    "box_layout": box_layout,
                    "keypose_path": [f"keypose{idx}_path" for idx in range(len(box_layout))],
                    "local_prompt": local_prompt,  # add local prompt to ease the sample selection
                }
            }
            compose_condition_config.update(this_layout)
        OmegaConf.save(OmegaConf.create(compose_condition_config), os.path.join(compose_cond_dir, f"compose_{args.cond_type}.yaml"))


    if args.update_final_gen_config:
        # save the config for final gen
        final_gen_config = {
            "num_pannels": len(global_prompt_list),
            "background_prompt": bg_prompt_list,
            "global_prompt": global_prompt_list,
            "negative_prompt": neg_prompt_list,
            "prompt_rewrite": panel_texts_list,
            f"{args.cond_type}_condition": condtional_file_path_list,  # path to the condition files
            f"{args.cond_type}_adaptor_weight": adaptor_weight_list,  # allow per-sample tuning
        }
        final_gen_dir = os.path.join(args.work_dir, "final_gen")
        os.makedirs(final_gen_dir, exist_ok=True)
        OmegaConf.save(OmegaConf.create(final_gen_config), os.path.join(final_gen_dir, "final_gen.yaml"))

