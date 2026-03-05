# !/bin/bash

python demo/image_demo.py \
    /home/ww/code/story/mix-of-show/Grounded-Segment-Anything/outputs/hina/1/raw_image.jpg \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    /nas2/wwen/weights/t2i-adaptor/T2I-Adapter/models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --out-file vis_results_org.jpg
