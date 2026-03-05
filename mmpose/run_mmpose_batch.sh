# !/bin/bash

python demo/image_demo_batch.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --input_dir "../Grounded-Segment-Anything/outputs/hina" \
    --image_type "image_crop.png" \
    --output_dir "outputs/hina" \
    --kpt-thr 0.3
