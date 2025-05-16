#!/bin/bash
set -x
RESULTS_PATH=../../../raw_results/image_classification/

# python3  classify.py --model_name google/vit-base-patch16-224 --results_path $RESULTS_PATH/vit-base-patch16-224.xlsx
# python3  classify.py --model_name google/vit-large-patch16-224 --results_path $RESULTS_PATH/vit-large-patch16-224.xlsx
# python3 classify.py --model_name timm/vit_huge_patch14_224_in21k --results_path $RESULTS_PATH/vit_huge_patch14_224_in21k.xlsx
# python3 classify.py --model_name timm/vit_large_patch16_224.augreg_in21k --results_path $RESULTS_PATH/vit_large_patch16_224.augreg_in21k.xlsx
# python3 classify.py --model_name timm/vit_large_patch16_224.augreg_in21k_ft_in1k --results_path $RESULTS_PATH/vit_large_patch16_224.augreg_in21k_ft_in1k.xlsx
# python3 classify.py --model_name timm/vit_large_patch14_clip_224.laion2b_ft_in1k  --results_path $RESULTS_PATH/vit_large_patch14_clip_224.laion2b_ft_in1k.xlsx
# python3 classify.py --model_name timm/davit_base.msft_in1k  --results_path $RESULTS_PATH/davit_base.msft_in1k.xlsx
# python3 classify.py --model_name timm/inception_next_base.sail_in1k --results_path $RESULTS_PATH/inception_next_base.sail_in1k.xlsx
# python3 classify.py --model_name timm/swinv2_base_window8_256.ms_in1k --results_path $RESULTS_PATH/swinv2_base_window8_256.ms_in1k.xlsx
# python3 classify.py --model_name timm/cspresnet50 --results_path $RESULTS_PATH/cspresnet50.xlsx
# python3 classify.py --model_name timm/cspresnext50 --results_path $RESULTS_PATH/cspresnext50.xlsx
# python3 classify.py --model_name timm/vgg16.tv_in1k --results_path ${RESULTS_PATH}/vgg16.tv_in1k.xlsx
# python3 classify.py --model_name timm/vgg19.tv_in1k --results_path $RESULTS_PATH/vgg19.tv_in1k.xlsx
# python3 classify.py --model_name timm/resnet50.tv_in1k --results_path $RESULTS_PATH/resnet50.tv_in1k.xlsx
# python3 classify.py --model_name timm/resnet101.tv_in1k --results_path $RESULTS_PATH/resnet101.tv_in1k.xlsx
# python3 classify.py --model_name timm/resnet152.tv_in1k --results_path $RESULTS_PATH/resnet152.tv_in1k.xlsx
# python3 classify.py --model_name timm/mobilenetv3_large_100.ra_in1k --results_path $RESULTS_PATH/mobilenetv3_large_100.ra_in1k.xlsx
# python3 classify.py --model_name timm/nasnetalarge.tf_in1k --results_path $RESULTS_PATH/nasnetalarge.tf_in1k.xlsx
# python3 classify.py --model_name timm/convnext_xlarge.fb_in22k_ft_in1k --results_path $RESULTS_PATH/convnext_xlarge.fb_in22k_ft_in1k.xlsx
# python3 classify.py --model_name timm/resnet50.fb_swsl_ig1b_ft_in1k --results_path $RESULTS_PATH/resnet50.fb_swsl_ig1b_ft_in1k.xlsx
# python3 classify.py --model_name timm/resnet50_clip.cc12m --results_path $RESULTS_PATH/resnet50_clip.cc12m.xlsx
# python3 classify.py --model_name timm/resnet50_clip.yfcc15m --results_path $RESULTS_PATH/resnet50_clip.yfcc15m.xlsx
# python3 classify.py --model_name timm/resnet50.fb_ssl_yfcc100m_ft_in1k  --results_path $RESULTS_PATH/resnet50.fb_ssl_yfcc100m_ft_in1k.xlsx

python3 classify.py --model_name timm/cspdarknet53.ra_in1k --results_path $RESULTS_PATH/cspdarknet53.ra_in1k.xlsx
# python3 classify.py --model_name timm/darknet53.c2ns_in1k --results_path $RESULTS_PATH/darknet53.c2ns_in1k.xlsx

# # this is the same model as google/siglip-large-patch16-384
# python3 classify.py --model_name openclip/ViT-L-16-SigLIP-384/webli --results_path $RESULTS_PATH/ViT-L-16-SigLIP-384.xlsx

# python3 classify.py --model_name microsoft/swinv2-base-patch4-window16-256 --results_path $RESULTS_PATH/swinv2-base-patch4-window16-256.xlsx
# python3 classify.py --model_name microsoft/swin-large-patch4-window12-384 --results_path $RESULTS_PATH/swin-large-patch4-window12-384.xlsx
# python3 classify.py --model_name microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft --results_path $RESULTS_PATH/swinv2-large-patch4-window12to24-192to384-22kto1k-ft.xlsx

# for MODEL_NAME in internimage_b_1k_224 internimage_l_22kto1k_384 internimage_xl_22kto1k_384 internimage_h_22to1k_640 internimage_g_22kto1k_512
# do
# 	python3  classify.py --model_name OpenGVLab/${MODEL_NAME} --results_path $RESULTS_PATH/${MODEL_NAME}.xlsx
# done

# python3 classify.py --model_name timm/eva_giant_patch14_224.clip_ft_in1k  --results_path $RESULTS_PATH/eva_giant_patch14_224.clip_ft_in1k.xlsxp

# python3 classify.py --model_name openai/clip-vit-large-patch14 --results_path $RESULTS_PATH/clip-vit-large-patch14.xlsx
# python3 classify.py --model_name openai/clip-vit-large-patch14-336 --results_path $RESULTS_PATH/clip-vit-large-patch14-336.xlsx
# python3 classify.py --model_name timm/eva02_large_patch14_448.mim_m38m_ft_in1k --results_path $RESULTS_PATH/eva02_large_patch14_224_mim_m38m.xlsx
# python3 classify.py --model_name google/siglip-so400m-patch14-384 --results_path $RESULTS_PATH/siglip-so400m-patch14-384.xlsx
# wrong class labels
# python3 classify.py --model_name google/siglip-large-patch16-384 --results_path $RESULTS_PATH/siglip-large-patch16-384.xlsx

# #large CNNs
# python3 classify.py --model_name timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k --results_path $RESULTS_PATH/convnext_base.clip_laion2b_augreg_ft_in12k_in1k.xlsx
# python3 classify.py --model_name timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k --results_path $RESULTS_PATH/convnext_xxlarge.clip_laion2b_soup_ft_in1k.xlsx
# python3 classify.py --model_name timm/convnext_large_mlp.clip_laion2b_augreg_ft_in1k --results_path $RESULTS_PATH/convnext_large_mlp.clip_laion2b_augreg_ft_in1k.xlsx

# python3  classify.py --model_name openclip/ViT-bigG-14/laion2b_s39b_b160k --results_path $RESULTS_PATH/ViT-bigG-14_laion2b_s39b_b160k.xlsx --gpu_device -1
 
# python3 classify.py --model_name openclip/EVA02-E-14/laion2b_s4b_b115k --results_path $RESULTS_PATH/EVA02-E-14_laion2b_s4b_b115k.xlsx
# python3 classify.py --model_name openclip/EVA02-E-14-plus/laion2b-s9b_b144k --results_path $RESULTS_PATH/EVA02-E-14-plus_laion2b-s9b_b144k.xlsx
# TODO figure out linear probing
# python3 classify.py --model_name OpenGVLab/InternViT-300M-448px --results_path $RESULTS_PATH/OpenGVLab/InternViT-300M-448px.xlsx
# python3 classify.py --model_name OpenGVLab/InternViT-6B-224px --results_path $RESULTS_PATH/OpenGVLab/InternViT-6B-224px.xlsx


# python3 classify.py --model_name timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k  --results_path $RESULTS_PATH/swin_large_patch4_window7_224_in22k_ft_in1k.xlsx
# python3 classify.py --model_name timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k  --results_path $RESULTS_PATH/swin_base_patch4_window7_224_in22k_ft_in1k.xlsx

# python3 classify.py --model_name vit_large_patch16_mae --results_path $RESULTS_PATH/vit_large_patch16_mae.xlsx
# python3 classify.py --model_name vit_base_patch16_mae --results_path $RESULTS_PATH/vit_base_patch16_mae.xlsx
# python3 classify.py --model_name vit_huge_patch14_mae --results_path $RESULTS_PATH/vit_huge_patch14_mae.xlsx

# python3 classify.py --model_name openclip/ViT-L-14-quickgelu/dfn2b --results_path $RESULTS_PATH/ViT-L-14-quickgelu_dfn2b.xlsx

# python3 classify.py --model_name openclip/ViT-H-14-quickgelu/dfn5b --results_path $RESULTS_PATH/ViT-H-14-quickgelu_dfn5b.xlsx --gpu_device -1
# python3 classify.py --model_name openclip/ViT-H-14-quickgelu/dfn5b --results_path $RESULTS_PATH/ViT-H-14-quickgelu_dfn5b.xlsx

# python3 classify.py --model_name openclip/ViT-H-14/dfn5b --results_path $RESULTS_PATH/ViT-H-14_dfn5b.xlsx --gpu_device -1
# python3 classify.py --model_name openclip/ViT-H-14/dfn5b --results_path $RESULTS_PATH/ViT-H-14_dfn5b.xlsx

# python3 classify.py --model_name openclip/ViT-L --results_path $RESULTS_PATH/DFN5B-CLIP-ViT-H-14.xlsx

# Run zero-shot models twice
# first time on CPU to generate all text ecodings
# second time on a GPU with image data

# python3 classify.py --model_name openclip/ViT-g-14/laion2b_s12b_b42k --results_path $RESULTS_PATH/ViT-g-14_laion2b_s12b_b42k.xlsx --gpu_device -1
# python3 classify.py --model_name openclip/ViT-g-14/laion2b_s12b_b42k --results_path $RESULTS_PATH/ViT-g-14_laion2b_s12b_b42k.xlsx

# python3 classify.py --model_name openclip/ViT-g-14/laion2b_s34b_b88k --results_path $RESULTS_PATH/ViT-g-14_laion2b_s34b_b88k.xlsx --gpu_device -1
# python3 classify.py --model_name openclip/ViT-g-14/laion2b_s34b_b88k --results_path $RESULTS_PATH/ViT-g-14_laion2b_s34b_b88k.xlsx

# python3 classify.py --model_name openclip/RN50-quickgelu/openai --results_path $RESULTS_PATH/RN50_quickgelu_openai.xlsx --gpu_device -1
# python3 classify.py --model_name openclip/RN50-quickgelu/openai --results_path $RESULTS_PATH/RN50_quickgelu_openai.xlsx

# python3 classify.py --model_name openclip/RN50-quickgelu/yfcc15m --results_path $RESULTS_PATH/RN50_quickgelu_yfcc15m.xlsx --gpu_device -1
# python3 classify.py --model_name openclip/RN50-quickgelu/yfcc15m --results_path $RESULTS_PATH/RN50_quickgelu_yfcc15m.xlsx

# python3 classify.py --model_name openclip/RN101-quickgelu/openai --results_path $RESULTS_PATH/RN101_quickgelu_openai.xlsx --gpu_device -1
# python3 classify.py --model_name openclip/RN101-quickgelu/openai --results_path $RESULTS_PATH/RN101_quickgelu_openai.xlsx

# python3 classify.py --model_name openclip/RN101-quickgelu/yfcc15m --results_path $RESULTS_PATH/RN101_quickgelu_yfcc15m.xlsx --gpu_device -1
# python3 classify.py --model_name openclip/RN101-quickgelu/yfcc15m --results_path $RESULTS_PATH/RN101_quickgelu_yfcc15m.xlsx
