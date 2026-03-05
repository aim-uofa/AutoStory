#!/bin/bash 

export CUDA_VISIBLE_DEVICES=0

# use relative path for ckpts, so the script can be shared for different users
python grounded_sam_demo_batch.py \
	--config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
	--grounded_checkpoint groundingdino_swint_ogc.pth \
	--sam_checkpoint sam_vit_h_4b8939.pth \
	--input_dir "../results/EDLoRA_hina_Anyv4_B4_Iter1K/visualization/PromptDataset/iters_EDLoRA_hina_Anyv4_B4_Iter1K_negprompt" \
	--output_dir "outputs/hina/" \
	--box_threshold 0.3 \
	--text_threshold 0.25 \
	--text_prompt "person" \
	--device "cuda"
