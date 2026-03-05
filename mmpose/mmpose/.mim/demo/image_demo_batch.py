# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

import os
import glob
import tqdm

import cv2
import numpy as np

skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

pose_kpt_color = [[51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0],
                  [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],
                  [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0]]

pose_link_color = [[0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0],
                   [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0], [255, 128, 0],
                   [0, 255, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
                   [51, 153, 255], [51, 153, 255], [51, 153, 255]]


def imshow_keypoints(img,
                     pose_result,
                     kpt_score_thr=0.1,
                     radius=2,
                     thickness=2):
    """Draw keypoints and links on an image.

    Args:
            img (ndarry): The image to draw poses on.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            thickness (int): Thickness of lines.
    """

    img_h, img_w, _ = img.shape
    img = np.zeros(img.shape)

    # for idx, kpts in enumerate(pose_result):
    #     if idx > 1:
    #         continue
    #     kpts = kpts['keypoints']
    #     # print(kpts)
    kpts = pose_result
    kpts = np.array(kpts, copy=False)

    # draw each point on image
    assert len(pose_kpt_color) == len(kpts)

    for kid, kpt in enumerate(kpts):
        x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

        if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
            # skip the point that should not be drawn
            continue

        color = tuple(int(c) for c in pose_kpt_color[kid])
        cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)

    # draw links
    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0 or pos1[1] >= img_h or pos2[0] <= 0
                or pos2[0] >= img_w or pos2[1] <= 0 or pos2[1] >= img_h or kpts[sk[0], 2] < kpt_score_thr
                or kpts[sk[1], 2] < kpt_score_thr or pose_link_color[sk_id] is None):
            # skip the link that should not be drawn
            continue
        color = tuple(int(c) for c in pose_link_color[sk_id])
        cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--input_dir', help='dir to grouding-dino outputs')
    parser.add_argument('--image_type', help='the type of image in grouding-dino outputs')
    parser.add_argument('--output_dir', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    args = parser.parse_args()
    return args


def main(args):

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    # inference a single image
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(args.img, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        out_file=args.out_file)

    # prepare input for visulization
    keypoints = results.pred_instances.keypoints
    keypoint_scores = results.pred_instances.keypoint_scores
    assert len(keypoints) == len(keypoint_scores) == 1
    keypoints = keypoints[0]
    keypoint_scores = keypoint_scores[0]
    keypoints_vis_input = np.concatenate([keypoints, keypoint_scores[:, None]], axis=1)

    pose = imshow_keypoints(img, keypoints_vis_input, radius=2, thickness=2, kpt_score_thr=args.kpt_thr)
    # save pose as image
    cv2.imwrite(args.out_file.replace('.png', '_pose.png'), pose)


if __name__ == '__main__':
    args = parse_args()

    input_dir_share = args.input_dir
    output_dir_share = args.output_dir
    image_folders = os.path.join(input_dir_share, "*", "*", "*")  # iters/prompt/image_name/
    for input_dir in glob.glob(image_folders):
        image_name = os.path.basename(input_dir)
        args.img = os.path.join(input_dir, args.image_type)
        args.out_file = args.img.replace(input_dir_share, output_dir_share)
        args.out_file = os.path.dirname(args.out_file)

        main(args)
