# CUDA_VISIBLE_DEVICES="2,3" torchrun \
# --nproc_per_node=2 --master_port=2234 mixofshow/test.py \
# -opt options/test/MixofShow/EDLoRA/characters/anime/EDLoRA_maki_Anyv4_B4_Iter1K.yml --launcher pytorch

CUDA_VISIBLE_DEVICES="3" python mixofshow/test.py -opt options/test/MixofShow/EDLoRA/objects/real/EDLoRA_dogB_Cmix_dropkv02_B2_Iter500.yml
