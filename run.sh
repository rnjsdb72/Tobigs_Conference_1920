# echo "Matting..."

# python ./Matting/inference.py \
#     --variant resnet50 \
#     --checkpoint "./ckpt/rvm_resnet50.pth" \
#     --device cpu \
#     --input-source "./img" \
#     --output-type png_sequence \
#     --output-composition "output" \
#     --output-alpha "output" \
#     --output-foreground "output" \
#     --output-video-mbps 4 \
#     --seq-chunk 1

# echo "Inpainting..."

# python ./Inpainting/inference_propainter.py \
#     --inputs ./img --input_type image\
#     --output ./output \
#     --height -1 --width -1

echo "Estimating Pose..."

python ./PoseEstimation/inference.py \
    --model-cfg ./PoseEstimation/configs/td-hm_3xrsn50_8xb32-210e_coco-256x192.py \
    --vis-cfg ./PoseEstimation/configs/coco.py \
    --ckpt ./ckpt/td-hm_3xrsn50_8xb32-210e_coco-256x192-c3e3c4fe_20221013.pth \
    --input-dir ./output --apply-matted "True" --output-dir ./output \
    --device cpu