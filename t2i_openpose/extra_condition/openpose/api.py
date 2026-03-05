import numpy as np
import os
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from PIL import Image
import torch

import glob
import tqdm
import sys
# sys.path.append('/home/ww/code/story/T2I-Adapter')
# sys.path.append('/home/ww/code/story/T2I-Adapter/ldm/modules/extra_condition/openpose')
from t2i_openpose.extra_condition.openpose import util
from t2i_openpose.extra_condition.openpose.body import Body

remote_model_path = "https://huggingface.co/TencentARC/T2I-Adapter/blob/main/third-party-models/body_pose_model.pth"


class OpenposeInference(nn.Module):

    def __init__(self, body_modelpath=None):
        super().__init__()
        if body_modelpath is None:
            body_modelpath = os.path.join('models', "body_pose_model.pth")

        if not os.path.exists(body_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir='models')

        self.body_estimation = Body(body_modelpath)

    def forward(self, x):
        x = x[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(x)
            canvas = np.zeros_like(x)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        return canvas


# def get_cond_openpose(opt, cond_image, cond_inp_type='image', cond_model=None):
#     if isinstance(cond_image, str):
#         openpose_keypose = cv2.imread(cond_image)
#     else:
#         openpose_keypose = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
#     openpose_keypose = resize_numpy_image(
#         openpose_keypose, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
#     opt.H, opt.W = openpose_keypose.shape[:2]
#     if cond_inp_type == 'openpose':
#         openpose_keypose = img2tensor(openpose_keypose).unsqueeze(0) / 255.
#         openpose_keypose = openpose_keypose.to(opt.device)
#     elif cond_inp_type == 'image':
#         with autocast('cuda', dtype=torch.float32):
#             openpose_keypose = cond_model(openpose_keypose)
#         openpose_keypose = img2tensor(openpose_keypose).unsqueeze(0) / 255.
#         openpose_keypose = openpose_keypose.to(opt.device)

#     else:
#         raise NotImplementedError

#     return openpose_keypose


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


if __name__ == "__main__":

    cond_image_dir = "/home/ww/code/story/mix-of-show/results/EDLoRA_hina_Anyv4_B4_Iter1K/visualization/PromptDataset/iters_EDLoRA_hina_Anyv4_B4_Iter1K_negprompt/"
    output_dir = "./outputs/detect-openpose"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda")
    body_modelpath = "/nas2/wwen/weights/t2i-adaptor/T2I-Adapter/models/body_pose_model.pth"

    model = OpenposeInference(body_modelpath=body_modelpath).to(device=device)
    for idx, cond_image_path in tqdm.tqdm(enumerate(glob.glob(os.path.join(cond_image_dir, "*.png")))):
        input_image = cv2.imread(cond_image_path)
        # openpose_keypose = resize_numpy_image(openpose_keypose, max_resolution=512 * 512, resize_short_edge=None)

        with torch.autocast('cuda', dtype=torch.float32):
            openpose_keypose = model(input_image)
        # openpose_keypose = img2tensor(openpose_keypose).unsqueeze(0) / 255.
        # openpose_keypose = openpose_keypose.to(device)

        # save image
        output_path_image = os.path.join(output_dir, f"{idx}_input.png")
        output_path_openpose = os.path.join(output_dir, f"{idx}_openpose.png")
        # save image using cv2, note openpose_keypose is in ndarray format
        cv2.imwrite(output_path_image, input_image)
        cv2.imwrite(output_path_openpose, openpose_keypose)
        # overlay openpose_keypose on image and save
        composed_image = cv2.addWeighted(input_image, 0.5, openpose_keypose, 0.5, 0)
        output_path_composed = os.path.join(output_dir, f"{idx}_composed.png")
        cv2.imwrite(output_path_composed, composed_image)
