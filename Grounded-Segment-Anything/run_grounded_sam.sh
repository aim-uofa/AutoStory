#!/bin/bash 

export CUDA_VISIBLE_DEVICES=1

text_prompt=(
	"dog" \
	"dog" \
	"cat" \
	"cat" \
	"dog" \
	"cat" \
	"dog" \
	"cat" \
	"dog" \
	"dog" \
	"cat" 
	)

input_image=(
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/EDLoRA_dogB_Cmix_dropkv02_B2_Iter500/visualization/PromptDataset/iters_EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt/A_walking_\<dogB1\>_\<dogB2\>---G_7.5_S_50---49---EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/EDLoRA_dogB_Cmix_dropkv02_B2_Iter500/visualization/PromptDataset/iters_EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt/A_\<dogB1\>_\<dogB2\>,_facing_left---G_7.5_S_50---20---EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/cata_test/visualization/PromptDataset/iters_cata_test_negprompt/An_angry_\<catA1\>_\<catA2\>,_arching_back---G_7.5_S_50---1---cata_test_negprompt.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/cata_test/visualization/PromptDataset/iters_cata_test_negprompt/cat.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/EDLoRA_dogB_Cmix_dropkv02_B2_Iter500/visualization/PromptDataset/iters_EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt/A_walking_\<dogB1\>_\<dogB2\>---G_7.5_S_50---49---EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/experiments/catA_edlora/visualization/PromptDataset/iters_501_negprompt/a_\<catA1\>_\<catA2\>_sit_on_the_chair---G_7.5_S_50---1---501_negprompt.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/experiments/dogb_edlora/visualization/PromptDataset/iters_501_negprompt/\<dogB1\>_\<dogB2\>---G_7.5_S_50---7---501_negprompt.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/cata_test/visualization/PromptDataset/iters_cata_test_negprompt/An_angry_\<catA1\>_\<catA2\>,_arching_back---G_7.5_S_50---47---cata_test_negprompt.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/EDLoRA_dogB_Cmix_dropkv02_B2_Iter500/visualization/PromptDataset/iters_EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt/A_\<dogB1\>_\<dogB2\>,_facing_left---G_7.5_S_50---12---EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt.png \
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/EDLoRA_dogB_Cmix_dropkv02_B2_Iter500/visualization/PromptDataset/iters_EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt/A_\<dogB1\>_\<dogB2\>,_sleeping---G_7.5_S_50---33---EDLoRA_dogB_Cmix_dropkv02_B2_Iter500_negprompt.png\
	/mnt/nas/share/home/canyu/Diffusion/Mix-of-Show/results/cata_test/visualization/PromptDataset/iters_cata_test_negprompt/A_\<catA1\>_\<catA2\>,_sleeping---G_7.5_S_50---32---cata_test_negprompt.png \
	)

output_dir=(
	"outputs/1" \
	"outputs/2" \
	"outputs/3" \
	"outputs/4" \
	"outputs/5" \
	"outputs/6" \
	"outputs/7" \
	"outputs/8" \
	"outputs/9" \
	"outputs/10" \
	"outputs/11" 
)

for((i=0;i<${#text_prompt[@]};i++));  
do   
	python grounded_sam_demo.py \
		--config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
		--grounded_checkpoint /mnt/nas/share/home/canyu/Diffusion/STORY/sam/groundingdino_swint_ogc.pth \
		--sam_checkpoint /mnt/nas/share/home/canyu/Diffusion/STORY/sam/sam_vit_h_4b8939.pth \
		--input_image ${input_image[i]} \
		--output_dir ${output_dir[i]} \
		--box_threshold 0.3 \
		--text_threshold 0.25 \
		--text_prompt ${text_prompt[i]} \
		--device "cuda"
done 

