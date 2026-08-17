cd /root/FastMRI_challenge

mkdir -p \
  /root/result/promptmr8_metric_aligned_50ep_v1

python3 -u train.py \
  -g 0 \
  -n promptmr8_metric_aligned_50ep_v1 \
  -e 50 \
  -b 1 \
  -r 100 \
  -l 2e-4 \
  -t /root/Data/train/ \
  -v /root/Data/val/ \
  --seed 430 \
  --num-cascades 8 \
  --n-history 3 \
  -c true \
  --compute-sens-per-coil true \
  --bbox-loss-weight 0.3 \
  --lr-scheduler true \
  --lr-milestones 16 27 \
  --lr-gamma 0.3 \
  --mask-aug true \
  --mask-aug-weight 1.0 \
  --mask-aug-start 16 \
  --mask-aug-schedule exp \
  --mask-aug-plateau-epoch 25 \
  --mask-aug-accelerations 4 8 \
  --mask-aug-random-ratio 0.0 \
  --mask-aug-random-offset true \
  --aug_on \
  --aug_schedule exp \
  --aug_delay 30 \
  --aug_strength 0.5 \
  --aug_exp_decay 5.0 \
  --aug_interpolation_order 1 \
  --aug_weight_translation 0.1 \
  --aug_weight_rotation 0.1 \
  --aug_weight_shearing 0.1 \
  --aug_weight_scaling 1.0 \
  --aug_weight_rot90 0.0 \
  --aug_weight_fliph 0.4 \
  --aug_weight_flipv 0.0 \
  --aug_max_translation_x 0.05 \
  --aug_max_translation_y 0.05 \
  --aug_max_rotation 10.0 \
  --aug_max_shearing_x 10.0 \
  --aug_max_shearing_y 10.0 \
  --aug_max_scaling 0.05 \
  --annealing-epoch 1 \
  2>&1 | tee \
  /root/result/promptmr8_metric_aligned_50ep_v1/train_stdout.log