#CUDA_VISIBLE_DEVICES=0 
python ../run_image_classification.py \
    --train_dir ../LessImagesTest/train \
    --validation_dir ../LessImagesTest/val \
    --output_dir ./norwegian_animals_outputs/ \
    --remove_unused_columns False \
    --label_column_name label \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --label_smoothing_factor 0.1 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
#    --model_name_or_path google/vit-base-patch16-224

# python ../run_image_classification.py \
#     --train_dir ../LessImagesTest/train \
#     --validation_dir ../LessImagesTest/val \
#     --output_dir ./norwegian_animals_outputs/ \
#     --remove_unused_columns False \
#     --label_column_name label \
#     --do_train \
#     --do_eval \
#     --learning_rate 2e-5 \
#     --num_train_epochs 20 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --eval_strategy epoch \
#     --save_strategy epoch \
#     --load_best_model_at_end True \
#     --save_total_limit 3 \
#     --seed 1337
 #   --overwrite_output_dir

