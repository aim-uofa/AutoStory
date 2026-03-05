#!/bin/bash 

sketch_condition=(
  "/mnt/nas/share/home/canyu/Diffusion/STORY/ours_region/story_utils/1.png" \
  "/mnt/nas/share/home/canyu/Diffusion/STORY/ours_region/story_utils/2.png" \
  "/mnt/nas/share/home/canyu/Diffusion/STORY/ours_region/story_utils/3.png" \
  "/mnt/nas/share/home/canyu/Diffusion/STORY/ours_region/story_utils/4.png" \
  "/mnt/nas/share/home/canyu/Diffusion/STORY/ours_region/story_utils/5.png" \
  "/mnt/nas/share/home/canyu/Diffusion/STORY/ours_region/story_utils/6.png" 
)

sketch_adaptor_weight=(
  1 \
  1 \
  1 \
  1 \
  1 \
  1 
)

context_prompt=(
  "A sunny afternoon, in the forest, a dog happily walking" \
  "A cat and a dog, in the forest" \
  "A cat and a dog, in the forest" \
  "A cat and a dog, in the forest" \
  "A cat and a dog playing a red ball in the forest." \
  "A cat and a dog sleeping together in the forest, sunshine in the background"
)

# write in this format
# "${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"
prompt_rewrite=(
  "a walking <dogB1> <dogB2>, joyful-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[185, 450, 445, 731]" \
  "a <catA1> <catA2> on the rock-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[10, 150, 512, 500]|a <dogB1> <dogB2> in the forest-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[10, 512, 512, 1024]" \
  "a <catA1> <catA2> in the forest-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[206, 386, 444, 640]|a <dogB1> <dogB2> in the forest-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[140, 600, 320, 950]" \
  "a <catA1> <catA2>-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[181, 190, 376, 480]|a happy <dogB1> <dogB2>-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[170, 630, 400, 900]" \
  "a <catA1> <catA2>-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[80, 234, 340, 620]|a red ball-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[274, 578, 339, 643]|a happy <dogB1> <dogB2>-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[100, 700, 464, 916]"\ 
  "a <dogB1> <dogB2>, sleeping-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[136, 120, 350, 500]|a <catA1> <catA2>, sleeping-*-longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality-*-[104, 510, 451, 822]"
)

expdir="dog_and_cat"

for((i=0;i<${#sketch_condition[@]};i++));  
do   
  # no key pose here
  python inference/mix_of_show_sample.py \
    --keypose_adaptor_model="/mnt/nas/share/home/canyu/Diffusion/checkpoints/t2iadapter/t2iadapter_openpose_sd14v1.pth" \
    --sketch_adaptor_model="/mnt/nas/share/home/canyu/Diffusion/checkpoints/t2iadapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight[i]}\
    --sketch_condition=${sketch_condition[i]} \
    --save_dir="results/multi-concept/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt[i]}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite[i]}" \
    --suffix="${i}" \
    --seed=55555
done
