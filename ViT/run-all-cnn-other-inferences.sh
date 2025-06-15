for model_path in \
  ../outputs/cnn-and-other-models-othersmall/convnext-base-384_lr1em04_seed42_ep30_bs8 \
  ../outputs/cnn-and-other-models-othersmall/beit-base-patch16-384_lr1em04_seed42_ep30_bs8 \
  ../outputs/cnn-and-other-models-othersmall/cvt-13-384_lr1em04_seed42_ep30_bs8 \
  ../outputs/cnn-and-other-models-othersmall/resnet-152_lr1em04_seed42_ep30_bs8 \
  ../outputs/cnn-and-other-models-othersmall/resnet-18_lr1em04_seed42_ep30_bs8 \
  ../outputs/cnn-and-other-models-othersmall/resnet-50_lr1em04_seed42_ep30_bs8
do
  echo "Running inference for: $model_path"
  python ../ViT/inference-cnn-other-combined-classes-gpu-classwise-statistics-and-confusion-matrix.py \
    --model-path "$model_path" \
    --image-folder ../nina_images/testsetCombinedClasses
done
